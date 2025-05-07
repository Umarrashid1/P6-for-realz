import os
import numpy as np
import pandas as pd
from config import numerical_columns  # List of your numeric feature columns

def compute_and_save_global_stats(dataset_dir, output_file):
    n_cols = len(numerical_columns)
    sum_vals = np.zeros(n_cols)
    sum_sqs = np.zeros(n_cols)
    count_vals = 0

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, engine="pyarrow", usecols=numerical_columns)
            except Exception as e:
                print(f"[SKIP] Error reading {file_path}: {e}")
                continue

            if df.empty:
                continue

            subset = df[numerical_columns].dropna().values
            if subset.size == 0:
                continue

            sum_vals += np.sum(subset, axis=0)
            sum_sqs += np.sum(subset ** 2, axis=0)
            count_vals += subset.shape[0]

    if count_vals == 0:
        raise RuntimeError("No valid numeric data found — check dataset or column names.")

    global_means = sum_vals / count_vals
    global_stds = np.sqrt((sum_sqs / count_vals) - (global_means ** 2)) + 1e-6

    # ✅ Save column names along with the stats
    np.savez(
        output_file,
        mean=global_means,
        std=global_stds,
        cols=np.array(numerical_columns)
    )

    print(f"[INFO] Saved global stats to {output_file}")
    print(f"[INFO] Columns: {numerical_columns}")
    print(f"[INFO] Mean: {global_means}")
    print(f"[INFO] Std: {global_stds}")

if __name__ == "__main__":
    compute_and_save_global_stats(
        dataset_dir="../../../dataset/raw_dataset",
        output_file="standardization_stats.npz"
    )
