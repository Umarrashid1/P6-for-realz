import os
import numpy as np
import pandas as pd
from pipeline.config import numerical_columns_flows, numerical_columns_packets  # User's import

IS_FLOW = False  # True for flow stats, False for packet stats


DATASET_DIR = "../../../dataset/roni/DatasetFlow" if IS_FLOW else "../../../dataset/raw_dataset"
OUTPUT_FILE = "flow_standardization_stats.npz" if IS_FLOW else "packet_standardization_stats.npz"


def compute_and_save_global_stats():
    numerical_columns = numerical_columns_flows if IS_FLOW else numerical_columns_packets
    all_data = []

    # Load and concatenate all data
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            try:
                # --- Minimal Change 1: Robust usecols ---
                df_header = pd.read_csv(file_path, nrows=0)
                cols_to_load = [col for col in numerical_columns if col in df_header.columns]
                if not cols_to_load:

                    continue
                df = pd.read_csv(file_path, engine="pyarrow", usecols=cols_to_load).astype("float32")
                # --- End Minimal Change 1 ---
            except Exception as e:
                print(f"[SKIP] Error reading {file_path}: {e}")
                continue

            if df.empty:
                continue

            # Add missing numeric columns (from full numerical_columns list) as NaNs
            for col in numerical_columns:
                if col not in df.columns:  # If it wasn't in cols_to_load initially
                    df[col] = np.nan

            df = df[numerical_columns]  # Ensure column order
            all_data.append(df)

    if not all_data:
        raise RuntimeError("No valid numeric data found — check dataset or column names.")

    df_full = pd.concat(all_data, ignore_index=True)

    # Handle infinities early
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)


    # 2️⃣ Clipping
    clip_low = df_full.quantile(0.001)  # quantile is NaN-safe
    clip_high = df_full.quantile(0.999)  # quantile is NaN-safe
    df_full = df_full.clip(lower=clip_low, upper=clip_high, axis=1)  # clip handles NaN bounds correctly

    # 3️⃣ Median imputation
    medians = df_full.median()  # pandas median skips NaNs by default
    # --- Minimal Change 3: Ensure medians used for filling are not NaN ---
    medians.fillna(0, inplace=True)  # If a column was all NaN, its median is NaN; fill that median with 0.
    # --- End Minimal Change 3 ---
    df_full = df_full.fillna(medians)  # Now all NaNs should be filled with a number or 0.

    # 4️⃣ Mean & Std
    arr = df_full.values  # df_full should now be clean of NaNs and Infs

    # Because df_full should be clean now, np.sum is expected to work.
    # If warnings still occur here, it means NaNs persisted through the fillna(medians) step,
    # which would imply a deeper issue or a column that was entirely NaN and whose median became 0.
    sum_vals = np.sum(arr, axis=0)
    sum_sqs = np.sum(arr ** 2, axis=0)
    count_vals = arr.shape[0]

    if count_vals == 0:
        raise RuntimeError("Array for mean/std calculation is empty (0 rows).")

    global_means = sum_vals / count_vals

    # --- Minimal Change 4: Robust variance calculation ---
    variance = (sum_sqs / count_vals) - (global_means ** 2)
    variance[variance < 0] = 0  # Prevent sqrt of negative due to precision issues
    global_stds = np.sqrt(variance)
    # --- End Minimal Change 4 ---
    global_stds[global_stds < 1e-6] = 1.0  # Adjusted threshold slightly, keeps original value for std replacement

    # 5️⃣ Save results
    # Ensure series passed to np.savez are correctly indexed and NaN-filled for saving
    clip_low_save = clip_low.reindex(numerical_columns).fillna(0).values
    clip_high_save = clip_high.reindex(numerical_columns).fillna(0).values
    medians_save = medians.reindex(numerical_columns).fillna(0).values

    np.savez(
        OUTPUT_FILE,
        mean=global_means,
        std=global_stds,
        cols=np.array(numerical_columns),
        clip_low=clip_low_save,
        clip_high=clip_high_save,
        median=medians_save,  # medians_save already had NaNs (from all-NaN columns) replaced with 0
    )

    print(f"[INFO] Saved global stats to {OUTPUT_FILE}")
    print(f"[INFO] Columns: {numerical_columns}")
    print(f"[INFO] Mean (first 5): {global_means[:5]}")  # Print sample
    print(f"[INFO] Std (first 5): {global_stds[:5]}")  # Print sample
    # print(f"[INFO] Clip low: {clip_low_save}") # Can be very long
    # print(f"[INFO] Clip high: {clip_high_save}")
    # print(f"[INFO] Medians: {medians_save}")


if __name__ == "__main__":
    compute_and_save_global_stats()