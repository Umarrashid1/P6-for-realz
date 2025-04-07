import os
import pandas as pd
import numpy as np
from .config import categorical_columns, numerical_columns, LABEL_MAPPING
import torch

# Define global variables for min/max of numerical columns
global_min = pd.Series(dtype='float64')
global_max = pd.Series(dtype='float64')


def first_pass(dataset_dir):
    global global_min, global_max
    print("Starting first pass (collecting stats)...")

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Update global min/max for numerical columns
                numeric_df = df[numerical_columns].select_dtypes(include=[np.number])
                global_min = pd.concat([global_min, numeric_df.min()]).groupby(level=0).min()
                global_max = pd.concat([global_max, numeric_df.max()]).groupby(level=0).max()


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


def second_pass(dataset_dir, output_file):
    print("Starting second pass (processing and writing to .pt)...")

    all_numerical = []
    all_categorical = []
    all_labels = []

    file_counter = 0
    skipped_files = 0

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                file_counter += 1
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                df = pd.read_csv(file_path)

                label = find_label_from_path(file_path)
                if label == -1:
                    print(f"[SKIP] Unknown folder: {file_path}")
                    skipped_files += 1
                    continue

                # Safety: check required columns exist
                if not all(col in df.columns for col in numerical_columns + categorical_columns):
                    print(f"[SKIP] Missing columns in: {file_path}")
                    skipped_files += 1
                    continue

                # Normalize numerical
                for col in numerical_columns:
                    if col in df.columns and col in global_min and col in global_max:
                        min_val, max_val = global_min[col], global_max[col]
                        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0

                # Convert to tensors
                df_numerical = df[numerical_columns].astype(np.float32)
                df_categorical = df[categorical_columns].astype("category").apply(lambda x: x.cat.codes).astype(
                    np.int64)
                labels = np.full(len(df), label, dtype=np.int64)

                all_numerical.append(torch.tensor(df_numerical.values))
                all_categorical.append(torch.tensor(df_categorical.values))
                all_labels.append(torch.tensor(labels))

    print(f"Finished. Processed files: {file_counter}, Skipped: {skipped_files}")

    if not all_numerical:
        raise RuntimeError("No valid data found. All files were skipped.")

    numerical_tensor = torch.cat(all_numerical, dim=0)
    categorical_tensor = torch.cat(all_categorical, dim=0)
    label_tensor = torch.cat(all_labels, dim=0)

    torch.save({
        "numerical": numerical_tensor,
        "categorical": categorical_tensor,
        "label": label_tensor
    }, output_file)

    print(f"Data saved as PyTorch tensor to {output_file}")
