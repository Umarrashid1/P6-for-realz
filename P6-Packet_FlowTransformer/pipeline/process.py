import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from typing import List, Tuple
import logging
import datetime


from .config import categorical_columns, numerical_columns, LABEL_MAPPING

# ── Set up logging ──────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"preprocessing_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)



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

def list_csv_files(dataset_dir: str) -> List[str]:
    return [
        f for f in glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        if not f.endswith(":Zone.Identifier")
    ]

def load_csv_file(file_path: str, test_mode: bool, rows_per_file: int):
    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            nrows=rows_per_file if test_mode and rows_per_file else None,
        )
        logging.info(f"[LOADED] {file_path} — shape: {df.shape}")  # ✱ LOGGING ADDED
        return df, file_path
    except Exception as e:
        logging.warning(f"[SKIP] Could not read {file_path}: {e}")  # ✱ LOGGING ADDED
        return None

def process_fragment(args) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    df, file_path, missing_strategy, max_seq_len = args

    label = find_label_from_path(file_path)
    if label == -1:
        logging.warning(f"[SKIP] No label for: {file_path}")  # ✱ LOGGING ADDED
        return [], [], []

    required_flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing_cols = [c for c in numerical_columns + categorical_columns if c not in df.columns]
    if missing_cols or any(c not in df.columns for c in required_flow_cols):
        logging.warning(f"[SKIP] Missing columns in {file_path}: {missing_cols}")  # ✱ LOGGING ADDED
        return [], [], []

    df["protocol"] = np.select(
        [df["l4_tcp"].eq(1), df["l4_udp"].eq(1)], ["TCP", "UDP"], default="OTHER"
    )

    df = df[
        numerical_columns
        + categorical_columns
        + ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    ].copy()

    # Fill missing values
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

    # Standardize
    for col in numerical_columns:
        df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)

    # Categorical encoding
    df[categorical_columns] = df[categorical_columns].astype("category").apply(
        lambda x: x.cat.codes
    )

    group_keys = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    flow_groups = df.groupby(group_keys, sort=False)

    flow_lengths = [len(g) for _, g in flow_groups]
    protocol_counts = df["protocol"].value_counts().to_dict()
    logging.info(f"[METRICS] {file_path} — {len(flow_lengths)} flows, avg len {np.mean(flow_lengths):.1f}, max len {np.max(flow_lengths)}, protocols: {protocol_counts}")  # ✱ LOGGING ADDED

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
            num_chunks = (flow_len - max_seq_len) // stride + 1
            for start in range(0, flow_len - max_seq_len + 1, stride):
                end = start + max_seq_len
                pkt_arrays.append(flow_features[start:end].astype(np.float32))
                mask_arrays.append(np.ones(max_seq_len, dtype=np.float32))
                label_list.append(label)
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
    file_paths = list_csv_files(dataset_dir)
    logging.info(f"[START] Found {len(file_paths)} CSV files.")  # ✱ LOGGING ADDED

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        load_futs = [
            tpool.submit(load_csv_file, file_path, test_mode, rows_per_file)
            for file_path in file_paths
        ]
        loaded = [f.result() for f in load_futs if f.result() is not None]

    if not loaded:
        raise RuntimeError("No files loaded (all skipped or failed).")

    logging.info(f"[LOAD COMPLETE] {len(loaded)} files successfully loaded.")  # ✱ LOGGING ADDED

    proc_args = [(df, path, missing_strategy, max_seq_len) for df, path in loaded]
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
        {"packet_seq": packet_tensor, "label": label_tensor, "attention_mask": attention_mask_tensor},
        output_file,
    )

    logging.info(f"[DONE] Saved {len(packet_tensor)} flows to {output_file}")  # ✱ LOGGING ADDED
    logging.info(f"[SHAPES] packets: {packet_tensor.shape}, labels: {label_tensor.shape}")  # ✱ LOGGING ADDED

    label_counts = np.bincount(lbl_list)
    for label_id, count in enumerate(label_counts):
        logging.info(f"[LABEL] class {label_id}: {count} sequences")  # ✱ LOGGING ADDED

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
