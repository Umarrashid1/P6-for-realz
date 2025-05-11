import pandas as pd
from pathlib import Path
import numpy as np  # For np.nan handling if needed, though pandas handles it well.


numerical_columns = [
    'inter_arrival_time', 'time_since_previously_displayed_frame', 'ttl', 'eth_size', 'tcp_window_size',
    'payload_entropy', 'handshake_cipher_suites_length', 'handshake_extensions_length', 'dns_len_qry',
    'dns_interval', 'dns_len_ans', 'payload_length', 'http_content_len', 'icmp_data_size',
    'jitter', 'stream_1_count', 'stream_1_mean', 'stream_1_var', 'src_ip_1_count', 'src_ip_1_mean', 'src_ip_1_var',
    'src_ip_mac_1_count', 'src_ip_mac_1_mean', 'src_ip_mac_1_var', 'channel_1_count', 'channel_1_mean', 'channel_1_var',
    'stream_jitter_1_sum', 'stream_jitter_1_mean', 'stream_jitter_1_var', 'stream_5_count', 'stream_5_mean',
    'stream_5_var', 'src_ip_5_count', 'src_ip_5_mean', 'src_ip_5_var', 'src_ip_mac_5_count', 'src_ip_mac_5_mean',
    'src_ip_mac_5_var', 'channel_5_count', 'channel_5_mean', 'channel_5_var', 'stream_jitter_5_sum',
    'stream_jitter_5_mean', 'stream_jitter_5_var', 'stream_10_count', 'stream_10_mean', 'stream_10_var',
    'src_ip_10_count', 'src_ip_10_mean', 'src_ip_10_var', 'src_ip_mac_10_count', 'src_ip_mac_10_mean',
    'src_ip_mac_10_var', 'channel_10_count', 'channel_10_mean', 'channel_10_var', 'stream_jitter_10_sum',
    'stream_jitter_10_mean', 'stream_jitter_10_var', 'stream_30_count', 'stream_30_mean', 'stream_30_var',
    'src_ip_30_count', 'src_ip_30_mean', 'src_ip_30_var', 'src_ip_mac_30_count', 'src_ip_mac_30_mean',
    'src_ip_mac_30_var', 'channel_30_count', 'channel_30_mean', 'channel_30_var', 'stream_jitter_30_sum',
    'stream_jitter_30_mean', 'stream_jitter_30_var', 'stream_60_count', 'stream_60_mean', 'stream_60_var',
    'src_ip_60_count', 'src_ip_60_mean', 'src_ip_60_var', 'src_ip_mac_60_count', 'src_ip_mac_60_mean',
    'src_ip_mac_60_var', 'channel_60_count', 'channel_60_mean', 'channel_60_var', 'stream_jitter_60_sum',
    'stream_jitter_60_mean', 'stream_jitter_60_var', 'ntp_interval', 'most_freq_spot', 'min_et', 'q1', 'min_e',
    'var_e', 'q1_e', 'sum_p', 'min_p', 'max_p', 'med_p', 'average_p', 'var_p', 'q3_p', 'q1_p', 'iqr_p', 'l3_ip_dst_count'
]

BASE_DIR = Path(__file__).resolve().parent  # Assumes script is in some project root
# Or define it absolutely if running from an arbitrary location:
# BASE_DIR = Path("/path/to/your/project_root_or_where_script_is")

DATASET_DIR = BASE_DIR / "../../dataset/raw_dataset"
OUTPUT_DIR = BASE_DIR / "../../dataset/standardized_dataset"

# Ensure the base output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using dataset directory: {DATASET_DIR.resolve()}")
print(f"Outputting to directory: {OUTPUT_DIR.resolve()}")
print(f"Numerical columns to standardize: {numerical_columns}")


# --- Helper Functions ---
def find_csv_files(directory):
    """Recursively finds all CSV files in a directory."""
    return list(Path(directory).rglob('*.csv'))


# --- Main Script ---

# Phase 1: Calculate global statistics (mean, std, median)
print("\nPhase 1: Calculating global statistics...")

all_numerical_data = []
input_files = find_csv_files(DATASET_DIR)

if not input_files:
    print(f"No CSV files found in {DATASET_DIR}. Exiting.")
    exit()

print(f"Found {len(input_files)} CSV files for processing.")

for i, file_path in enumerate(input_files):
    print(f"  Reading file {i + 1}/{len(input_files)}: {file_path.name}")
    try:
        df_chunk = pd.read_csv(file_path, usecols=numerical_columns, na_values=['', 'None', 'nan', 'NaN', 'NULL'])
        all_numerical_data.append(df_chunk)
    except Exception as e:
        print(f"    Error reading {file_path}: {e}. Skipping this file for stats calculation.")
        # You might want to handle this more robustly, e.g., by trying different encodings or logging errors.

if not all_numerical_data:
    print("No data could be read from any CSV files. Exiting.")
    exit()

# Concatenate all data for global calculations
global_df = pd.concat(all_numerical_data, ignore_index=True)
print(f"Successfully concatenated data from {len(all_numerical_data)} files. Total rows: {len(global_df)}")

# Calculate global medians for imputation
print("\nCalculating global medians for imputation...")
global_medians = {}
for col in numerical_columns:
    if col in global_df.columns:
        global_medians[col] = global_df[col].median()
        print(f"  Median for '{col}': {global_medians[col]}")
    else:
        print(f"  Warning: Column '{col}' not found in concatenated data. Cannot calculate median.")
        # Decide how to handle this: error, skip, or use a default (e.g., 0)
        # For now, it will lead to an error later if we try to access global_medians[col]
        # and the column was missing everywhere.

# Impute missing values in the concatenated DataFrame using global medians (for accurate mean/std)
print("\nImputing missing values globally before calculating mean/std...")
for col in numerical_columns:
    if col in global_df.columns and col in global_medians:
        original_nan_count = global_df[col].isnull().sum()
        global_df[col] = global_df[col].fillna(global_medians[col])
        filled_nan_count = original_nan_count - global_df[col].isnull().sum()
        if original_nan_count > 0:
            print(f"  Imputed {filled_nan_count} NaN values in '{col}' using median {global_medians[col]}.")
    elif col not in global_df.columns:
        print(f"  Skipping imputation for '{col}' as it was not found in any file.")

# Calculate global means and standard deviations
print("\nCalculating global means and standard deviations...")
global_means = {}
global_stds = {}

for col in numerical_columns:
    if col in global_df.columns:
        global_means[col] = global_df[col].mean()
        global_stds[col] = global_df[col].std()
        # Handle cases where standard deviation might be zero (e.g., constant column after imputation)
        if global_stds[col] == 0:
            print(f"  Warning: Standard deviation for '{col}' is 0. Standardized values will be 0.")
        print(f"  Mean for '{col}': {global_means[col]}, Std for '{col}': {global_stds[col]}")
    else:
        print(f"  Warning: Column '{col}' not found. Cannot calculate mean/std.")

# Phase 2: Apply transformations and save files
print("\nPhase 2: Applying transformations and saving files...")

for i, original_file_path in enumerate(input_files):
    relative_path = original_file_path.relative_to(DATASET_DIR)
    output_file_path = OUTPUT_DIR / relative_path

    # Create subdirectory structure in output if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Processing file {i + 1}/{len(input_files)}: {original_file_path.name} -> {output_file_path}")

    try:
        # Read the original file again, this time all columns
        df_to_transform = pd.read_csv(original_file_path, na_values=['', 'None', 'nan', 'NaN', 'NULL'])

        # Keep track of columns that were actually processed for logging
        processed_cols_in_file = []

        for col in numerical_columns:
            if col in df_to_transform.columns:
                processed_cols_in_file.append(col)
                # 1. Impute missing values using pre-calculated global medians
                if col in global_medians:
                    nan_count_before = df_to_transform[col].isnull().sum()
                    df_to_transform[col] = df_to_transform[col].fillna(global_medians[col])
                    if nan_count_before > 0:
                        # print(f"    Imputed NaNs in '{col}' using median {global_medians[col]}")
                        pass  # Reduce verbosity here, already logged globally
                else:
                    # This case should ideally not happen if the column was in numerical_columns
                    # and present in at least one file during Phase 1.
                    print(
                        f"    Warning: Global median not found for '{col}'. NaNs might persist or cause errors if not handled.")

                # 2. Standardize the column
                if col in global_means and col in global_stds:
                    if global_stds[col] != 0:
                        df_to_transform[col] = (df_to_transform[col] - global_means[col]) / global_stds[col]
                    else:
                        # If std is 0, it means all values (after imputation) are the same as the mean.
                        # So, (X - mean) is 0. The standardized value should be 0.
                        df_to_transform[col] = 0.0
                else:
                    print(f"    Warning: Global mean/std not found for '{col}'. Cannot standardize.")
            else:
                # This can happen if a numerical column listed in config is not in THIS specific file
                # print(f"    Column '{col}' not found in {original_file_path.name}. Skipping.")
                pass

        # Save the transformed DataFrame
        df_to_transform.to_csv(output_file_path, index=False)
        # print(f"    Saved standardized file to: {output_file_path}")
        if not processed_cols_in_file:
            print(
                f"    Warning: No numerical columns specified in `numerical_columns` were found in {original_file_path.name}.")


    except Exception as e:
        print(f"    Error processing or saving {original_file_path}: {e}")

print("\nDataset standardization complete.")
print(f"Original dataset: {DATASET_DIR.resolve()}")
print(f"Standardized dataset saved to: {OUTPUT_DIR.resolve()}")