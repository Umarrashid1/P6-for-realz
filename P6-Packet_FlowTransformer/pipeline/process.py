import os
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import datetime
from .config import categorical_columns, numerical_columns, LABEL_MAPPING
from utils import io_utils, category_mapping

# ── logging setup ─────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"preprocessing_{timestamp}.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── checkpoint dir ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints") / timestamp
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── load global stats (and prepare robust params) ─────────────────────────────
STATS = np.load("standardization_stats.npz")
GLOBAL_MEAN: Dict[str, float] = dict(zip(STATS["cols"].tolist(), STATS["mean"]))
GLOBAL_STD:  Dict[str, float] = dict(zip(STATS["cols"].tolist(), STATS["std"]))
EPS = 1e-6

# ── helper: robust clip + median fill consistent with stats script ────────────

def _robust_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to 0.1 % / 99.9 % and median‑fill per numeric column."""
    quantiles = df.quantile([0.001, 0.999])
    df = df.clip(lower=quantiles.loc[0.001], upper=quantiles.loc[0.999], axis=1)
    return df.fillna(df.median())

# ── core processing fn ────────────────────────────────────────────────────────

def process_fragment(args) -> Tuple[str, List[np.ndarray], List[int], List[np.ndarray], Dict[str, List[np.ndarray]]]:
    df, file_path, missing_strategy, max_seq_len = args

    # 0. find label -----------------------------------------------------------------
    def _find_label(p: str) -> int:
        cur = os.path.dirname(p)
        while cur and cur != os.path.dirname(cur):
            name = os.path.basename(cur)
            for k, v in LABEL_MAPPING.items():
                if k.lower() in name.lower():
                    return v
            cur = os.path.dirname(cur)
        return -1
    label = _find_label(file_path)
    if label == -1:
        logging.warning(f"[SKIP] No label for {file_path}")
        return file_path, [], [], [], {}

    # 1. required columns -----------------------------------------------------------
    required_flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
    missing_cols = [c for c in numerical_columns + categorical_columns if c not in df.columns]
    if missing_cols or any(c not in df.columns for c in required_flow_cols):
        logging.warning(f"[SKIP] Missing cols in {file_path}: {missing_cols}")
        return file_path, [], [], [], {}

    # 2. engineer 'protocol' categorical -------------------------------------------
    df["protocol"] = np.select(
        [df["l4_tcp"].eq(1), df["l4_udp"].eq(1)], ["TCP", "UDP"], default="OTHER"
    )

    df = df[numerical_columns + categorical_columns + required_flow_cols + ["protocol"]].copy()

    # 3. numeric cleaning (clip+median) --------------------------------------------
    df[numerical_columns] = _robust_clean(df[numerical_columns])

    # 4. fill categorical NaNs ------------------------------------------------------
    if missing_strategy == "zero":
        df[categorical_columns] = df[categorical_columns].fillna("unknown")
    elif missing_strategy in {"mean", "median", "ffill"}:
        # numeric part already handled; just categorical
        for col in categorical_columns:
            if missing_strategy == "ffill":
                df[col].fillna(method="ffill", inplace=True)
            else:
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                df[col].fillna(mode, inplace=True)
    else:
        raise ValueError("Unknown missing_strategy")

    # 5. standardise ---------------------------------------------------------------
    for col in numerical_columns:
        df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)
    # safety clip to ±10 σ
    df[numerical_columns] = df[numerical_columns].clip(-10, 10)

    # 6. map categoricals -----------------------------------------------------------
    cat_map = category_mapping.load_mappings()
    for col in categorical_columns:
        m = cat_map[col]
        unk = m["unknown"]
        df[col] = df[col].map(m).fillna(unk).astype(int)

    # 7. group flows & pad ----------------------------------------------------------
    group_keys = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    pkt_arrays, label_list, mask_arrays = [], [], []
    cat_arrays: Dict[str, List[np.ndarray]] = {c: [] for c in categorical_columns}

    for _, flow_df in df.groupby(group_keys, sort=False):
        if flow_df.empty:
            continue  # skip zero‑packet flows
        num_feat = flow_df[numerical_columns].values.astype(np.float32)
        cat_feat = {c: flow_df[c].values.astype(np.int64) for c in categorical_columns}
        L = len(num_feat)

        def add_slice(start, end):
            slice_len = end - start
            if slice_len == 0:
                return
            num_slice = num_feat[start:end]
            pad_len = max_seq_len - slice_len
            if pad_len > 0:
                num_slice = np.concatenate([num_slice, np.zeros((pad_len, len(numerical_columns)), np.float32)])
            pkt_arrays.append(num_slice)
            mask = np.concatenate([np.ones(slice_len), np.zeros(pad_len)], dtype=np.float32)
            mask_arrays.append(mask)
            label_list.append(label)
            for c in categorical_columns:
                unk = cat_map[c]["unknown"]
                cat_slice = cat_feat[c][start:end]
                if pad_len > 0:
                    cat_slice = np.pad(cat_slice, (0, pad_len), constant_values=unk)
                cat_arrays[c].append(cat_slice.astype(np.int64))

        if L <= max_seq_len:
            add_slice(0, L)
        else:
            stride = max_seq_len // 2
            for s in range(0, L - max_seq_len + 1, stride):
                add_slice(s, s + max_seq_len)
            if (L - max_seq_len) % stride:
                add_slice(L - max_seq_len, L)

    return file_path, pkt_arrays, label_list, mask_arrays, cat_arrays
