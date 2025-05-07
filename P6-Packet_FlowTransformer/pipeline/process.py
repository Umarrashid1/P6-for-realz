import os
import numpy as np
import pandas as pd
import torch
import pyarrow.dataset as ds
import pyarrow.csv as pv
import concurrent.futures
from typing import List, Tuple

from .config import categorical_columns, numerical_columns, LABEL_MAPPING

# ── LOAD GLOBAL NUMERIC STATS ───────────────────────────────────────────────
STD_STATS_PATH = "standardization_stats.npz"
_std_stats = np.load(STD_STATS_PATH)
STD_COLS = _std_stats["cols"].tolist()
GLOBAL_MEAN = dict(zip(STD_COLS, _std_stats["mean"]))
GLOBAL_STD = dict(zip(STD_COLS, _std_stats["std"]))
EPS = 1e-6  # numerical safety

# ---------------------------------------------------------------------------
# ❶  Helpers
# ---------------------------------------------------------------------------

csv_format = ds.CsvFileFormat(
    read_options=pv.ReadOptions(autogenerate_column_names=False)
)


def load_fragment(fragment, dataset_dir: str, test_mode: bool, rows_per_file: int):
    """I/O‑bound loader: returns (DataFrame, full_path) or None."""
    try:
        table = fragment.to_table()
        if test_mode and rows_per_file:
            table = table.slice(0, rows_per_file)
        df = table.to_pandas()
        return df, os.path.join(dataset_dir, fragment.path)
    except Exception as e:
        print(f"[SKIP] Could not read {fragment.path}: {e}")
        return None


def process_fragment(args) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    """CPU‑bound per‑file preprocessing executed in a *separate process*."""
    df, file_path, missing_strategy, max_seq_len = args

    label = find_label_from_path(file_path)
    if label == -1:
        print(f"[SKIP] No label for: {file_path}")
        return [], [], []

    required_flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing_cols = [c for c in numerical_columns + categorical_columns if c not in df.columns]
    if missing_cols or any(c not in df.columns for c in required_flow_cols):
        print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
        return [], [], []

    # protocol inference
    df["protocol"] = np.select(
        [df["l4_tcp"].eq(1), df["l4_udp"].eq(1)], ["TCP", "UDP"], default="OTHER"
    )

    df = df[
        numerical_columns
        + categorical_columns
        + ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    ].copy()

    # missing‑value handling
    if missing_strategy == "mean":
        for col in numerical_columns:
            df[col].fillna(df[col].mean(), inplace=True)
        for col in categorical_columns:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
    elif missing_strategy == "median":
        for col in numerical_columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_columns:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
    elif missing_strategy == "zero":
        df[numerical_columns] = df[numerical_columns].fillna(0)
        df[categorical_columns] = df[categorical_columns].fillna("unknown")
    elif missing_strategy == "ffill":
        df.fillna(method="ffill", inplace=True)
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # standardise
    for col in numerical_columns:
        df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)

    # categorical codes
    df[categorical_columns] = df[categorical_columns].astype("category").apply(
        lambda x: x.cat.codes
    )

    group_keys = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    flow_groups = df.groupby(group_keys, sort=False)

    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []

    for _, flow_df in flow_groups:
        flow_features = flow_df[numerical_columns + categorical_columns].values
        flow_len = len(flow_features)

        if flow_len < max_seq_len:
            pad_len = max_seq_len - flow_len
            attention = np.concatenate([np.ones(flow_len), np.zeros(pad_len)])
            pad = np.zeros((pad_len, flow_features.shape[1]), dtype=np.float32)
            pkt = np.concatenate([flow_features, pad], axis=0)
            pkt_arrays.append(pkt.astype(np.float32))
            mask_arrays.append(attention.astype(np.float32))
            label_list.append(label)
        else:
            stride = max_seq_len // 2
            for start in range(0, flow_len - max_seq_len + 1, stride):
                end = start + max_seq_len
                pkt_arrays.append(flow_features[start:end].astype(np.float32))
                mask_arrays.append(np.ones(max_seq_len, dtype=np.float32))
                label_list.append(label)
            # remainder
            if (flow_len - max_seq_len) % stride != 0:
                pkt_arrays.append(flow_features[-max_seq_len:].astype(np.float32))
                mask_arrays.append(np.ones(max_seq_len, dtype=np.float32))
                label_list.append(label)

    return pkt_arrays, label_list, mask_arrays

# ---------------------------------------------------------------------------
# ❷  Main driver
# ---------------------------------------------------------------------------

def preprocess_flows_as_sequences(
    dataset_dir,
    output_file,
    test_mode=False,
    rows_per_file=20000,
    missing_strategy="zero",
    max_seq_len=64,
):
    # 1) Discover Arrow fragments (files) ----------------------------------
    dataset = ds.dataset(dataset_dir, format=csv_format)

    # 2) Parallel I/O load (threads) ---------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        load_futs = [
            tpool.submit(load_fragment, frag, dataset_dir, test_mode, rows_per_file)
            for frag in dataset.get_fragments()
        ]
        loaded = [f.result() for f in load_futs if f.result() is not None]

    if not loaded:
        raise RuntimeError("No fragments loaded (all skipped or failed).")

    # 3) Parallel CPU preprocessing (processes) ----------------------------
    proc_args = [
        (df, path, missing_strategy, max_seq_len) for df, path in loaded
    ]

    pkt_list: List[np.ndarray] = []
    lbl_list: List[int] = []
    msk_list: List[np.ndarray] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ppool:
        for pkt_arrs, lbls, masks in ppool.map(process_fragment, proc_args, chunksize=1):
            pkt_list.extend(pkt_arrs)
            lbl_list.extend(lbls)
            msk_list.extend(masks)

    if not pkt_list:
        raise RuntimeError("No flows found after preprocessing.")

    packet_tensor = torch.tensor(np.stack(pkt_list), dtype=torch.float32)
    label_tensor = torch.tensor(lbl_list, dtype=torch.long)
    attention_mask_tensor = torch.tensor(np.stack(msk_list), dtype=torch.float32)

    torch.save(
        {
            "packet_seq": packet_tensor,
            "label": label_tensor,
            "attention_mask": attention_mask_tensor,
        },
        output_file,
    )

    print(
        f"\n[INFO] Preprocessing complete — saved {len(packet_tensor)} flows to {output_file}"
    )
    print(
        f"[INFO] Shape: packets {packet_tensor.shape}, labels {label_tensor.shape}"
    )


# ---------------------------------------------------------------------------
# ❸  Label helper (unchanged)
# ---------------------------------------------------------------------------

def find_label_from_path(file_path):
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1
