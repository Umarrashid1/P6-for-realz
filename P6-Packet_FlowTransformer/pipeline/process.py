import os
import pandas as pd
import numpy as np
import torch
from .config import LABEL_MAPPING

def preprocess_all_in_memory(dataset_dir,
                             output_file,
                             categorical_columns,
                             numerical_columns,
                             test_mode=False,
                             rows_per_file=2000,
                             missing_strategy="zero",
                             standardize=False):
    all_dfs = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                label = find_label_from_path(file_path)

                if label == -1:
                    print(f"[SKIP] No label for: {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path, nrows=rows_per_file if test_mode else None)
                except Exception as e:
                    print(f"[ERROR] Couldn't read {file_path}: {e}")
                    continue

                # Ensure all required columns are present
                missing_cols = [col for col in numerical_columns + categorical_columns if col not in df.columns]
                if missing_cols:
                    print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
                    continue

                df = df[numerical_columns + categorical_columns].copy()

                # Convert numeric columns explicitly
                for col in numerical_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df["__label__"] = label
                df["__source__"] = file_path
                all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No valid files found.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Replace inf values with NaN (not handled by fillna)
    full_df[numerical_columns] = full_df[numerical_columns].replace([np.inf, -np.inf], np.nan)

    # Debug: check for all-NaN numerical columns
    for col in numerical_columns:
        if full_df[col].isna().all():
            print(f"[WARNING] All values are NaN in column: {col}")

    # Handle missing values
    if missing_strategy == "mean":
        for col in numerical_columns:
            full_df[col].fillna(full_df[col].mean(), inplace=True)
        for col in categorical_columns:
            full_df[col].fillna(full_df[col].mode().iloc[0], inplace=True)

    elif missing_strategy == "median":
        for col in numerical_columns:
            full_df[col].fillna(full_df[col].median(), inplace=True)
        for col in categorical_columns:
            full_df[col].fillna(full_df[col].mode().iloc[0], inplace=True)

    elif missing_strategy == "zero":
        full_df[numerical_columns] = full_df[numerical_columns].fillna(0)
        full_df[categorical_columns] = full_df[categorical_columns].fillna("unknown")

    elif missing_strategy == "ffill":
        full_df.fillna(method='ffill', inplace=True)

    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # Final NaN check before scaling
    if full_df[numerical_columns].isna().any().any():
        print("[DEBUG] Columns with NaNs before scaling:")
        print(full_df[numerical_columns].isna().sum()[full_df[numerical_columns].isna().sum() > 0])
        raise ValueError("❌ NaNs still present after fillna!")

    # Scale numeric columns
    if standardize:
        print("[INFO] Using standardization (mean=0, std=1)")
        means = full_df[numerical_columns].mean()
        stds = full_df[numerical_columns].std()

        if means.isna().any() or stds.isna().any():
            print("[DEBUG] NaNs in mean or std detected before scaling")
            print("Means with NaNs:\n", means[means.isna()])
            print("Stds with NaNs:\n", stds[stds.isna()])
            raise ValueError("❌ NaNs in mean or std values before scaling")

        stds = stds.replace(0, 1).fillna(1)
        full_df[numerical_columns] = (full_df[numerical_columns] - means) / stds
    else:
        print("[INFO] Using min-max normalization")
        min_vals = full_df[numerical_columns].min()
        max_vals = full_df[numerical_columns].max()
        denom = max_vals - min_vals
        denom = denom.replace(0, 1).fillna(1)
        full_df[numerical_columns] = (full_df[numerical_columns] - min_vals) / denom

    # Final NaN check before saving
    if full_df[numerical_columns].isna().any().any():
        print("[DEBUG] Columns with NaNs after scaling:")
        print(full_df[numerical_columns].isna().sum()[full_df[numerical_columns].isna().sum() > 0])
        raise ValueError("❌ NaNs detected in numerical data after scaling!")

    # Encode categoricals
    full_df[categorical_columns] = full_df[categorical_columns].astype("category").apply(lambda x: x.cat.codes)

    # Print label distribution
    print("\n[INFO] Label distribution:")
    label_counts = full_df["__label__"].value_counts().sort_index()
    for label_id, count in label_counts.items():
        label_name = [k for k, v in LABEL_MAPPING.items() if v == label_id]
        label_str = label_name[0] if label_name else str(label_id)
        print(f"  {label_str} ({label_id}): {count} rows")

    # Convert to tensors
    numerical_tensor = torch.tensor(full_df[numerical_columns].values, dtype=torch.float32)
    categorical_tensor = torch.tensor(full_df[categorical_columns].values, dtype=torch.int64)
    label_tensor = torch.tensor(full_df["__label__"].values, dtype=torch.int64)

    torch.save({
        "numerical": numerical_tensor,
        "categorical": categorical_tensor,
        "label": label_tensor
    }, output_file)

    print(f"\n✅ Preprocessing complete — saved {len(full_df)} rows to {output_file}")


def find_label_from_path(file_path):
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1
