import os
import json
import pandas as pd
import concurrent.futures
from pathlib import Path
from . import io_utils
from pipeline.config import categorical_columns


def generate_and_save_category_mappings(
    dataset_dir: str,
    test_mode: bool = False,
    rows_per_file: int = 20_000,
    path: str = "category_mappings.json",
):
    all_files = io_utils.list_csv_files(dataset_dir)

    # ── 1. Load CSVs concurrently ──────────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        dfs = [
            f.result()
            for f in [pool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file) for fp in all_files]
            if f.result() is not None
        ]

    if not dfs:
        raise RuntimeError("No CSVs could be loaded to build mappings.")

    combined_df = pd.concat([df for df, _ in dfs], ignore_index=True)

    # ── 2. Build mappings column‑wise ──────────────────────────────────────
    mappings: dict[str, dict[str, int]] = {}
    for col in categorical_columns:
        raw_vals = combined_df[col].dropna().astype(str).unique().tolist()

        # Ensure "unknown" is **always** last
        raw_vals = [v for v in raw_vals if v.lower() != "unknown"]
        raw_vals = sorted(raw_vals)
        raw_vals.append("unknown")

        mapping = {val: idx for idx, val in enumerate(raw_vals)}
        mappings[col] = mapping

        # Sanity check: contiguous & unknown == last
        assert mapping["unknown"] == max(mapping.values()), f"{col}: unknown index mismatch"

        print(f"[MAPPING] {col}: {len(mapping)} categories (incl. 'unknown')")

    # ── 3. Save to disk ────────────────────────────────────────────────────
    Path(path).write_text(json.dumps(mappings, indent=2))
    return mappings


def load_mappings(path="category_mappings.json"):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    generate_and_save_category_mappings(dataset_dir="../../dataset/raw_dataset")  # Removed extra quote