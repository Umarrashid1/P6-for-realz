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
from .config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from utils import io_utils
from utils import category_mapping

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
GLOBAL_MEDIAN = dict(zip(STD_COLS, _std_stats["median"]))
CLIP_LOW = dict(zip(STD_COLS, _std_stats["clip_low"]))
CLIP_HIGH = dict(zip(STD_COLS, _std_stats["clip_high"]))
EPS = 1e-6

# ──────────────────────────────────────────────────────────────────────────────

def process_fragment(args) -> Tuple[str, List[np.ndarray], List[int], List[np.ndarray], dict]:
    df, file_path, max_seq_len = args
    label = find_label_from_path(file_path)
    if label == -1:
        logging.warning(f"[SKIP] No label for: {file_path}")
        return file_path, [], [], [], {}

    required_flow_cols = ["stream"]
    required_cols = numerical_columns_packets + categorical_columns_packets + required_flow_cols

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logging.warning(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
        return file_path, [], [], [], {}

    df = df[required_cols].copy()


    # Clip, fill, and standardize using global stats
    for col in numerical_columns_packets:
        df[col] = df[col].clip(lower=CLIP_LOW[col], upper=CLIP_HIGH[col])
        df[col] = df[col].fillna(GLOBAL_MEDIAN[col])
        df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)

    df[categorical_columns_packets] = df[categorical_columns_packets].fillna("unknown")

    cat_mappings = category_mapping.load_mappings()
    for col in categorical_columns_packets:
        mapping = cat_mappings[col]
        unknown_id = mapping["unknown"]
        df[col] = df[col].apply(lambda x: mapping.get(x, unknown_id)).astype(int)


    flow_groups = df.groupby("stream", sort=False)

    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    cat_arrays: dict[str, List[np.ndarray]] = {col: [] for col in categorical_columns_packets}

    for _, flow_df in flow_groups:
        num_feat = flow_df[numerical_columns_packets].values
        cat_feat = {col: flow_df[col].values for col in categorical_columns_packets}
        flow_len = len(num_feat)

        def pad_and_append(start_idx, end_idx):
            num_slice = num_feat[start_idx:end_idx]
            pkt_arrays.append(num_slice.astype(np.float32))
            mask_arrays.append(np.ones(max_seq_len, dtype=np.float32))
            label_list.append(label)
            for col in categorical_columns_packets:
                cat_arrays[col].append(cat_feat[col][start_idx:end_idx].astype(np.int64))

        if flow_len < max_seq_len:
            pad_len = max_seq_len - flow_len
            pkt = np.concatenate([num_feat, np.zeros((pad_len, len(numerical_columns_packets)), dtype=np.float32)], axis=0)
            pkt_arrays.append(pkt)
            mask_arrays.append(np.concatenate([np.ones(flow_len), np.zeros(pad_len)]).astype(np.float32))
            label_list.append(label)
            for col in categorical_columns_packets:
                unknown_id = cat_mappings[col]["unknown"]
                padded = np.pad(cat_feat[col], (0, pad_len), constant_values=unknown_id)
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
    max_seq_len: int = 64,
):
    all_files = io_utils.list_csv_files(dataset_dir)
    logging.info(f"[START] Found {len(all_files)} CSV files.")

    pending = [f for f in all_files if not (CHECKPOINT_DIR / (Path(f).stem + ".pt")).exists()]
    logging.info(f"[CHECKPOINT] {len(pending)} files pending processing.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        loaded = [f.result() for f in [tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file) for fp in pending] if f.result() is not None]

    if not loaded and not list(CHECKPOINT_DIR.glob("*.pt")):
        raise RuntimeError("No files loaded and no checkpoints found.")

    proc_args = [(df, path, max_seq_len) for df, path in loaded]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ppool:
        for file_path, pkt_arrs, lbls, masks, cat_arrs in ppool.map(process_fragment, proc_args, chunksize=1):
            if not pkt_arrs:
                continue
            shard = {
                "packet_seq": torch.tensor(np.stack(pkt_arrs), dtype=torch.float32),
                "label": torch.tensor(lbls, dtype=torch.long),
                "attention_mask": torch.tensor(np.stack(masks), dtype=torch.float32),
            }
            for col in categorical_columns_packets:
                shard[col] = torch.tensor(np.stack(cat_arrs[col]), dtype=torch.long)
            torch.save(shard, CHECKPOINT_DIR / (Path(file_path).stem + ".pt"))
            logging.info(f"[CKPT] wrote {file_path}")

    pkt_tensors, lbl_tensors, msk_tensors = [], [], []
    cat_tensors = {col: [] for col in categorical_columns_packets}

    for sf in sorted(CHECKPOINT_DIR.glob("*.pt")):
        data = torch.load(sf)
        pkt_tensors.append(data["packet_seq"])
        lbl_tensors.append(data["label"])
        msk_tensors.append(data["attention_mask"])
        for col in categorical_columns_packets:
            cat_tensors[col].append(data[col])

    output = {
        "packet_seq": torch.cat(pkt_tensors, dim=0),
        "label": torch.cat(lbl_tensors, dim=0),
        "attention_mask": torch.cat(msk_tensors, dim=0),
    }
    for col in categorical_columns_packets:
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