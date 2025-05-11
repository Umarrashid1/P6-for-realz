import os
import numpy as np
import pandas as pd
from .config import numerical_columns  # List of your numeric feature columns

def compute_and_save_global_stats(dataset_dir, output_file):
    all_data = []

    # 1️⃣ Load and concatenate all data
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, engine="pyarrow", usecols=numerical_columns).astype("float32")
            except Exception as e:
                print(f"[SKIP] Error reading {file_path}: {e}")
                continue

            if df.empty:
                continue

            # Add missing numeric columns as NaNs
            missing = [col for col in numerical_columns if col not in df.columns]
            for col in missing:
                df[col] = np.nan

            df = df[numerical_columns]  # Ensure column order
            all_data.append(df)

    if not all_data:
        raise RuntimeError("No valid numeric data found — check dataset or column names.")

    df_full = pd.concat(all_data, ignore_index=True)

    # 2️⃣ Compute global clipping bounds
    clip_low = df_full.quantile(0.001)
    clip_high = df_full.quantile(0.999)
    df_full = df_full.clip(lower=clip_low, upper=clip_high, axis=1)

    # 3️⃣ Compute and apply global medians
    medians = df_full.median()
    df_full = df_full.fillna(medians)

    # 4️⃣ Compute global mean and std
    arr = df_full.values
    sum_vals = np.sum(arr, axis=0)
    sum_sqs = np.sum(arr ** 2, axis=0)
    count_vals = arr.shape[0]

    global_means = sum_vals / count_vals
    global_stds = np.sqrt((sum_sqs / count_vals) - (global_means ** 2))
    global_stds[global_stds < 1e-2] = 1.0  # Prevent instability

    # 5️⃣ Save everything
    np.savez(
        output_file,
        mean=global_means,
        std=global_stds,
        cols=np.array(numerical_columns),
        clip_low=clip_low.values,
        clip_high=clip_high.values,
        median=medians.values,
    )

    print(f"[INFO] Saved global stats to {output_file}")
    print(f"[INFO] Columns: {numerical_columns}")
    print(f"[INFO] Mean: {global_means}")
    print(f"[INFO] Std: {global_stds}")
    print(f"[INFO] Clip low: {clip_low.values}")
    print(f"[INFO] Clip high: {clip_high.values}")
    print(f"[INFO] Medians: {medians.values}")

if __name__ == "__main__":
    compute_and_save_global_stats(
        dataset_dir="../../dataset/raw_dataset",
        output_file="standardization_stats.npz"
    )