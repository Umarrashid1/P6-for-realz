import os
import pandas as pd
import numpy as np
from .config import categorical_columns, numerical_columns, LABEL_MAPPING

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
    print("Starting second pass (processing and writing to CSV)...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    first_file = True

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Get label from any folder in the path
                label = find_label_from_path(file_path)
                if label == -1:
                    print(f"Skipping unknown folder: {file_path}")
                    continue

                # Normalize numerical columns
                for col in numerical_columns:
                    if col in df.columns and col in global_min and col in global_max:
                        min_val, max_val = global_min[col], global_max[col]
                        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0

                # Keep categorical columns as they are (for later embedding)
                df_cat = df[categorical_columns].copy()
                df = df.drop(columns=categorical_columns)

                # Reattach categorical columns
                df = pd.concat([df, df_cat], axis=1)

                # Add the label column
                df["label"] = label

                # Write DataFrame to CSV
                if first_file:
                    df.to_csv(output_file, mode='w', index=False)
                    first_file = False
                else:
                    df.to_csv(output_file, mode='a', header=False, index=False)

    print(f"CSV file written to {output_file}")
