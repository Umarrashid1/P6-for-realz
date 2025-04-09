import os
import pandas as pd
import numpy as np
from .config import categorical_columns, numerical_columns, LABEL_MAPPING
import torch

# Define global variables for min/max of numerical columns
global_min = pd.Series(dtype='float64')
global_max = pd.Series(dtype='float64')


def preprocess_all_in_memory(dataset_dir, output_file, test_mode=False, rows_per_file=100):
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

                missing_cols = [col for col in numerical_columns + categorical_columns if col not in df.columns]
                if missing_cols:
                    print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
                    continue

                df = df[numerical_columns + categorical_columns].copy()
                df["__label__"] = label
                df["__source__"] = file_path  # For diagnostics
                all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No valid files found.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Identify and print dropped rows
    before_drop = len(full_df)
    full_df["__drop_reason__"] = full_df[numerical_columns + categorical_columns].isnull().any(axis=1)
    dropped_rows = full_df[full_df["__drop_reason__"]]
    for _, row in dropped_rows.iterrows():
        print(f"[DROP] Row from {row['__source__']} had missing values: {row.to_dict()}")
    full_df.dropna(subset=numerical_columns + categorical_columns, inplace=True)

    # Normalize numerical
    full_df[numerical_columns] = (full_df[numerical_columns] - full_df[numerical_columns].min()) / (
        full_df[numerical_columns].max() - full_df[numerical_columns].min()
    )

    # Encode categorical
    full_df[categorical_columns] = full_df[categorical_columns].astype("category").apply(lambda x: x.cat.codes)

    # Print label distribution
    label_counts = full_df["__label__"].value_counts().sort_index()
    print("\n[INFO] Label distribution:")
    for label_id, count in label_counts.items():
        label_name = [k for k, v in LABEL_MAPPING.items() if v == label_id]
        label_str = label_name[0] if label_name else str(label_id)
        print(f"  {label_str} ({label_id}): {count} rows")

    # Convert to tensors
    numerical_tensor = torch.tensor(full_df[numerical_columns].values, dtype=torch.float32)
    categorical_tensor = torch.tensor(full_df[categorical_columns].values, dtype=torch.int64)
    label_tensor = torch.tensor(full_df["__label__"].values, dtype=torch.int64)

    # Save
    torch.save({
        "numerical": numerical_tensor,
        "categorical": categorical_tensor,
        "label": label_tensor
    }, output_file)

    print(f"\n[INFO] Preprocessing complete — saved {len(full_df)} rows to {output_file}")


def find_label_from_path(file_path):
    # Walk up the folder tree to find a matching label from LABEL_MAPPING
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):  # Stop at filesystem root
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1

