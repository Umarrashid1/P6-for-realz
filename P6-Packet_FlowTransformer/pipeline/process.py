import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple
import logging
import datetime
from .config import categorical_columns, numerical_columns, LABEL_MAPPING
from ..utils import io_utils
from ..utils import category_mapping




# ── Set up timestamp and logging ──────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"preprocessing_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Create a timestamped checkpoint directory ──────────────────────────────
CHECKPOINT_DIR = Path("checkpoints") / timestamp
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load global numeric stats ───────────────────────────────────────────────
STD_STATS_PATH = "standardization_stats.npz"
_std_stats = np.load(STD_STATS_PATH)
STD_COLS = _std_stats["cols"].tolist()
GLOBAL_MEAN = dict(zip(STD_COLS, _std_stats["mean"]))
GLOBAL_STD = dict(zip(STD_COLS, _std_stats["std"]))
EPS = 1e-6  # Numerical safety

# ──────────────────────────────────────────────────────────────────────────────


def process_fragment(args) -> Tuple[str, List[np.ndarray], List[int], List[np.ndarray], dict]:
    df, file_path, missing_strategy, max_seq_len = args
    label = find_label_from_path(file_path)
    if label == -1:
        logging.warning(f"[SKIP] No label for: {file_path}")
        return file_path, [], [], [], {}

    required_flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing_cols = [c for c in numerical_columns + categorical_columns if c not in df.columns]
    if missing_cols or any(c not in df.columns for c in required_flow_cols):
        logging.warning(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
        return file_path, [], [], []

    df["protocol"] = np.select(
        [df["l4_tcp"].eq(1), df["l4_udp"].eq(1)], ["TCP", "UDP"], default="OTHER"
    )

    df = df[
        numerical_columns + categorical_columns +
        ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
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

    # Standardize numerical features
    for col in numerical_columns:
        df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)

    cat_mappings = category_mapping.load_mappings()
    for col in categorical_columns:
        mapping = cat_mappings[col]
        unknown_id = mapping["unknown"]  # guaranteed to exist if you built the mappings correctly
        df[col] = df[col].map(mapping).fillna(unknown_id).astype(int)

    group_keys = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    flow_groups = df.groupby(group_keys, sort=False)

    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    cat_arrays: dict[str, List[np.ndarray]] = {col: [] for col in categorical_columns}

    for _, flow_df in flow_groups:
        num_feat = flow_df[numerical_columns].values
        cat_feat = {col: flow_df[col].values for col in categorical_columns}
        flow_len = len(num_feat)

        def pad_and_append(start_idx, end_idx):
            num_slice = num_feat[start_idx:end_idx]
            pkt_arrays.append(num_slice.astype(np.float32))
            mask_arrays.append(np.ones(max_seq_len, dtype=np.float32))
            label_list.append(label)
            for col in categorical_columns:
                cat_arrays[col].append(cat_feat[col][start_idx:end_idx].astype(np.int64))

        if flow_len < max_seq_len:
            pad_len = max_seq_len - flow_len
            pkt = np.concatenate([num_feat, np.zeros((pad_len, len(numerical_columns)), dtype=np.float32)], axis=0)
            pkt_arrays.append(pkt)
            mask_arrays.append(np.concatenate([np.ones(flow_len), np.zeros(pad_len)]).astype(np.float32))
            label_list.append(label)
            for col in categorical_columns:
                padded = np.pad(cat_feat[col], (0, pad_len), constant_values=-1)
                cat_arrays[col].append(padded.astype(np.int64))
        else:
            stride = max_seq_len // 2
            for start in range(0, flow_len - max_seq_len + 1, stride):
                pad_and_append(start, start + max_seq_len)
            if (flow_len - max_seq_len) % stride != 0:
                pad_and_append(flow_len - max_seq_len, flow_len)

    return file_path, pkt_arrays, label_list, mask_arrays, cat_arrays

# ──────────────────────────────────────────────────────────────────────────────

def preprocess_flows_as_sequences(
    dataset_dir: str,
    output_file: str,
    test_mode: bool = False,
    rows_per_file: int = 20000,
    missing_strategy: str = "zero",
    max_seq_len: int = 64,
):
    # 1) list and load
    all_files = io_utils.list_csv_files(dataset_dir)
    logging.info(f"[START] Found {len(all_files)} CSV files.")

    pending = [f for f in all_files if not (CHECKPOINT_DIR / (Path(f).stem + ".pt")).exists()]
    logging.info(f"[CHECKPOINT] {len(pending)} files pending processing.")

    # 3) load CSVs in threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        loaded = [f.result() for f in [tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file) for fp in pending] if f.result() is not None]

    if not loaded and not list(CHECKPOINT_DIR.glob("*.pt")):
        raise RuntimeError("No files loaded and no checkpoints found.")




    proc_args = [(df, path, missing_strategy, max_seq_len) for df, path in loaded]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ppool:
        for file_path, pkt_arrs, lbls, masks, cat_arrs in ppool.map(process_fragment, proc_args, chunksize=1):
            if not pkt_arrs:
                continue
            shard = {
                "packet_seq": torch.tensor(np.stack(pkt_arrs), dtype=torch.float32),
                "label": torch.tensor(lbls, dtype=torch.long),
                "attention_mask": torch.tensor(np.stack(masks), dtype=torch.float32),
            }
            for col in categorical_columns:
                shard[col] = torch.tensor(np.stack(cat_arrs[col]), dtype=torch.long)
            torch.save(shard, CHECKPOINT_DIR / (Path(file_path).stem + ".pt"))
            logging.info(f"[CKPT] wrote {file_path}")

    # Merge
    pkt_tensors, lbl_tensors, msk_tensors = [], [], []
    cat_tensors = {col: [] for col in categorical_columns}

    for sf in sorted(CHECKPOINT_DIR.glob("*.pt")):
        data = torch.load(sf)
        pkt_tensors.append(data["packet_seq"])
        lbl_tensors.append(data["label"])
        msk_tensors.append(data["attention_mask"])
        for col in categorical_columns:
            cat_tensors[col].append(data[col])

    output = {
        "packet_seq": torch.cat(pkt_tensors, dim=0),
        "label": torch.cat(lbl_tensors, dim=0),
        "attention_mask": torch.cat(msk_tensors, dim=0),
    }
    for col in categorical_columns:
        output[col] = torch.cat(cat_tensors[col], dim=0)

    torch.save(output, output_file)
    logging.info(f"[DONE] Merged dataset saved to {output_file}")

    label_counts = np.bincount(output["label"].numpy())
    for lbl_id, cnt in enumerate(label_counts):
        logging.info(f"[LABEL] class {lbl_id}: {cnt} seqs")

# ──────────────────────────────────────────────────────────────────────────────

def find_label_from_path(file_path: str) -> int:
    current_path = os.path.dirname(file_path)
    while current_path and current_path != os.path.dirname(current_path):
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1
