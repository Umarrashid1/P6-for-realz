import os
import json
import logging
import pandas as pd
import concurrent.futures
from pathlib import Path
import io_utils
from ..pipeline.config import categorical_columns


def generate_and_save_category_mappings(dataset_dir: str, test_mode: bool = False, rows_per_file: int = 20000, path="category_mappings.json"):
    all_files = io_utils.list_csv_files(dataset_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        loaded = [
            f.result()
            for f in [tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file) for fp in all_files]
            if f.result() is not None
        ]

    if not loaded:
        raise RuntimeError("No CSVs could be loaded to build mappings.")

    combined_df = pd.concat([df for df, _ in loaded], ignore_index=True)
    mappings = {}
    for col in categorical_columns:
        raw_vals = combined_df[col].dropna().unique().tolist()
        str_vals = sorted(str(v) for v in raw_vals)
        if "unknown" not in str_vals:
            str_vals.append("unknown")
        mappings[col] = {val: idx for idx, val in enumerate(str_vals)}

        logging.info(f"[MAPPING] {col}: {len(mappings[col])} categories (including 'unknown')")

    with open(path, "w") as f:
        json.dump(mappings, f)

    return mappings


def load_category_mappings(path="category_mappings.json"):
    with open(path, "r") as f:
        return json.load(f)


generate_and_save_category_mappings(dataset_dir="../../../dataset/raw_dataset'")