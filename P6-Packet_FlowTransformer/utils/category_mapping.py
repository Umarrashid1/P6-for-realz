import os
import json
import pandas as pd
import concurrent.futures
from pathlib import Path
import ast
import io_utils
from pipeline.config import categorical_columns_flows, categorical_columns_packets

# 🔧 TOGGLE THIS ONLY
IS_FLOW = False  # Set to True for flow mappings, False for packet mappings

# Auto-set dataset path and output path
DATASET_DIR = "../../dataset/roni/DatasetFlow" if IS_FLOW else "../../dataset/raw_dataset"
OUTPUT_PATH = "category_mappings_flows.json" if IS_FLOW else "category_mappings_packets.json"

def generate_and_save_category_mappings():
    categorical_columns = categorical_columns_flows if IS_FLOW else categorical_columns_packets
    all_files = io_utils.list_csv_files(DATASET_DIR)

    # ── 1. Load CSVs concurrently ──────────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        dfs = [
            f.result()
            for f in [pool.submit(io_utils.load_csv_file, fp, False, 20_000) for fp in all_files]
            if f.result() is not None
        ]

    if not dfs:
        raise RuntimeError("No CSVs could be loaded to build mappings.")

    combined_df = pd.concat([df for df, _ in dfs], ignore_index=True)

    # ── 2. Build mappings column‑wise ──────────────────────────────────────
    mappings: dict[str, dict[str, int]] = {}
    for col in categorical_columns:
        raw_vals = combined_df[col].dropna().unique().tolist()

        # Remove 'unknown' if present, then append explicitly
        raw_vals = [v for v in raw_vals if str(v).lower() != "unknown"]
        raw_vals = sorted(raw_vals, key=lambda x: str(x))
        raw_vals.append("unknown")

        mapping = {repr(val): idx for idx, val in enumerate(raw_vals)}
        mappings[col] = mapping

        assert repr("unknown") in mapping and mapping[repr("unknown")] == len(raw_vals) - 1, \
            f"{col}: 'unknown' index mismatch"

        print(f"[MAPPING] {col}: {len(mapping)} categories (incl. 'unknown')")

    # ── 3. Save mappings ───────────────────────────────────────────────────
    Path(OUTPUT_PATH).write_text(json.dumps(mappings, indent=2))
    print(f"[INFO] Saved category mappings to {OUTPUT_PATH}")
    return mappings


def load_mappings(path: str = None, is_flow: bool = None):
    if path is None:
        if is_flow is None:
            raise ValueError("Must provide either 'path' or 'is_flow'.")
        path = "category_mappings_flows.json" if is_flow else "category_mappings_packets.json"

    with open(path, "r") as f:
        raw = json.load(f)

    return {
        col: {ast.literal_eval(k): v for k, v in mapping.items()}
        for col, mapping in raw.items()
    }


if __name__ == "__main__":
    generate_and_save_category_mappings()
