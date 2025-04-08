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
                df = pd.read_csv(file_path, low_memory=False)

                # Update global min/max for numerical columns
                numeric_df = df[numerical_columns].select_dtypes(include=[np.number])
                global_min = pd.concat([global_min, numeric_df.min()]).groupby(level=0).min()
                global_max = pd.concat([global_max, numeric_df.max()]).groupby(level=0).max()

    print("\n[DEBUG] Global min/max sample:")
    print(global_min.head(5))
    print(global_max.head(5))


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
    skipped_label = 0
    skipped_columns = 0
    total_skipped = 0

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                file_counter += 1
                file_path = os.path.join(root, file)
                print(f"\n[INFO] Processing file: {file_path}")

                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"[ERROR] Could not read file: {e}")
                    total_skipped += 1
                    continue

                label = find_label_from_path(file_path)
                if label == -1:
                    print(f"[SKIP] No label match for path: {file_path}")
                    skipped_label += 1
                    total_skipped += 1
                    continue

                missing_num = [col for col in numerical_columns if col not in df.columns]
                missing_cat = [col for col in categorical_columns if col not in df.columns]

                if missing_num or missing_cat:
                    print(f"[SKIP] Missing columns in {file_path}")
                    if missing_num:
                        print(f"  Missing numerical columns: {missing_num[:5]}{'...' if len(missing_num) > 5 else ''}")
                    if missing_cat:
                        print(f"  Missing categorical columns: {missing_cat[:5]}{'...' if len(missing_cat) > 5 else ''}")
                    skipped_columns += 1
                    total_skipped += 1
                    continue

                # Normalize numerical features
                for col in numerical_columns:
                    if col in global_min and col in global_max:
                        min_val, max_val = global_min[col], global_max[col]
                        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0

                df_numerical = df[numerical_columns].astype(np.float32)
                df_categorical = df[categorical_columns].astype("category").apply(lambda x: x.cat.codes).astype(np.int64)
                labels = np.full(len(df), label, dtype=np.int64)

                all_numerical.append(torch.tensor(df_numerical.values))
                all_categorical.append(torch.tensor(df_categorical.values))
                all_labels.append(torch.tensor(labels))

    print("\n[SUMMARY]")
    print(f"  Total CSV files found:      {file_counter}")
    print(f"  Files skipped (no label):   {skipped_label}")
    print(f"  Files skipped (bad cols):   {skipped_columns}")
    print(f"  Total skipped:              {total_skipped}")
    print(f"  Successfully processed:     {file_counter - total_skipped}")

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

    print(f"\n✅ Data saved to {output_file}")
