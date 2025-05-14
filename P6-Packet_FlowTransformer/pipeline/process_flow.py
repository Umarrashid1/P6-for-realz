# pipeline/preprocess_flows.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional  # Added Dict
import logging
import datetime

# --- Configuration ---
# Attempt to import flow-specific columns first, then fall back or error
try:
    from config import categorical_columns_flows, numerical_columns_flows, LABEL_MAPPING

    print("✅ Using flow-specific columns from .config")
except ImportError:
    print("⚠️ '.config' not found or flow-specific columns missing. Attempting 'pipeline.config'.")
    try:
        from pipeline.config import categorical_columns_flows, numerical_columns_flows, LABEL_MAPPING

        print("✅ Using flow-specific columns from pipeline.config")
    except ImportError:
        print("❌ CRITICAL: Could not import flow-specific column lists or LABEL_MAPPING.")
        print("   Ensure 'categorical_columns_flows', 'numerical_columns_flows', and 'LABEL_MAPPING' are defined.")
        exit(1)
    except AttributeError:
        print("❌ CRITICAL: 'categorical_columns_flows' or 'numerical_columns_flows' not defined in config.")
        exit(1)

# Assuming utils are in a directory accessible from where this script is run
# (e.g., if this script is in 'pipeline/', and 'utils/' is a sibling or in PYTHONPATH)
try:
    from .utils import io_utils
    from .utils import category_mapping
except ImportError:
    print("❌ CRITICAL: Could not import 'utils.io_utils' or 'utils.category_mapping'.")
    print("   Ensure the 'utils' directory is correctly placed and importable.")
    exit(1)

# ── Set up timestamp and logging ──────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"preprocessing_flows_{timestamp}.log"  # Changed log file name

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("🚀 Starting Flow Data Preprocessing Script 🚀")

# ── Create a timestamped checkpoint directory ──────────────────────────────
CHECKPOINT_DIR = Path("checkpoints_flows") / timestamp  # Changed checkpoint dir name
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

# ── Load global numeric stats ───────────────────────────────────────────────
STD_STATS_PATH = "standardization_stats_flows.npz"  # This path should ideally point to FLOW-specific stats
logging.warning("🚨 CRITICAL WARNING: Using existing 'standardization_stats.npz'. 🚨")
logging.warning("   This file was likely generated for PACKET features, not FLOW features.")
logging.warning("   Applying these stats to flow numerical features WILL LIKELY lead to incorrect scaling.")
logging.warning("   It is STRONGLY recommended to generate and use standardization stats")
logging.warning("   derived specifically from your FLOW numerical features for this script to be correct.")
try:
    _std_stats = np.load(STD_STATS_PATH)
    STD_COLS = _std_stats["cols"].tolist()
    GLOBAL_MEAN = dict(zip(STD_COLS, _std_stats["mean"]))
    GLOBAL_STD = dict(zip(STD_COLS, _std_stats["std"]))
    GLOBAL_MEDIAN = dict(zip(STD_COLS, _std_stats["median"]))
    CLIP_LOW = dict(zip(STD_COLS, _std_stats["clip_low"]))
    CLIP_HIGH = dict(zip(STD_COLS, _std_stats["clip_high"]))
    EPS = 1e-6
    logging.info(f"Loaded standardization stats from {STD_STATS_PATH} (intended for packets).")
except FileNotFoundError:
    logging.error(f"❌ Standardization stats file not found: {STD_STATS_PATH}. Cannot proceed with standardization.")
    raise  # Stop if stats are missing and standardization is expected
except Exception as e:
    logging.error(f"❌ Error loading standardization stats: {e}")
    raise


# ──────────────────────────────────────────────────────────────────────────────

def process_single_flow_file(args: Tuple[pd.DataFrame, str]) -> Tuple[
    str, Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
    """
    Processes a DataFrame where each row is a flow.
    Applies standardization and categorical encoding.
    Returns numerical data, categorical data, and labels as NumPy arrays/lists.
    """
    df, file_path = args
    # Assuming one label per file for simplicity, as in the original script
    label = find_label_from_path(file_path)
    if label == -1:
        # logging.warning(f"[SKIP FILE] No label for: {file_path}") # Reduced logging
        return file_path, None, None, None

    # Use flow-specific column lists
    required_cols = numerical_columns_flows + categorical_columns_flows
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        # logging.warning(f"[SKIP FILE] Missing columns in {file_path}: {missing_cols}")
        return file_path, None, None, None

    df_processed = df[required_cols].copy()

    # Clip, fill, and standardize numerical features using global stats
    # ⚠️ This uses packet-derived stats. See CRITICAL WARNING above.
    for col in numerical_columns_flows:
        if col in df_processed.columns:
            if col not in GLOBAL_MEAN:  # Check if stats exist for this flow column
                logging.warning(
                    f"No standardization stats for flow numerical column '{col}'. Skipping its standardization.")
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)  # Basic fill
                continue
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].clip(lower=CLIP_LOW[col], upper=CLIP_HIGH[col])
            df_processed[col] = df_processed[col].fillna(GLOBAL_MEDIAN[col])
            df_processed[col] = (df_processed[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)
        else:  # Should not happen due to missing_cols check, but as safety
            logging.error(f"Numerical column '{col}' missing during processing for {file_path}.")
            return file_path, None, None, None

    # Fill NaN in categorical columns with "unknown" string before mapping
    df_processed[categorical_columns_flows] = df_processed[categorical_columns_flows].fillna("unknown").astype(str)

    # Apply category mappings
    try:
        # Ensure category_mappings.json is for FLOW features or is general enough
        cat_mappings = category_mapping.load_mappings()  # Assumes default path "category_mappings.json"
        for col in categorical_columns_flows:
            if col in df_processed.columns:
                if col not in cat_mappings:
                    logging.warning(f"No mapping for flow cat column '{col}'. Encoding on-the-fly.")
                    df_processed[col] = pd.Categorical(df_processed[col]).codes
                    continue

                mapping = cat_mappings[col]
                # Determine unknown_id carefully based on how load_mappings returns keys
                unknown_id = mapping.get("unknown",
                                         mapping.get(repr("unknown")))  # Try string "unknown" then repr("unknown")
                if unknown_id is None:  # Fallback if "unknown" not in map
                    # logging.warning(f"'unknown' not in mapping for {col}. Using 0 for unmapped.")
                    unknown_id = 0

                # Original script used .apply(lambda x: mapping.get(x, unknown_id))
                # This assumes keys in mapping are the original values.
                # load_mappings from your util returns original values as keys.
                df_processed[col] = df_processed[col].apply(lambda x: mapping.get(x, unknown_id)).astype(int)
            else:  # Should not happen
                logging.error(f"Categorical column '{col}' missing during processing for {file_path}.")
                return file_path, None, None, None

    except FileNotFoundError:
        logging.error(f"Category mappings file not found. Cannot encode.")
        raise
    except Exception as e:
        logging.error(f"Error applying category mappings for {file_path}: {e}")
        raise

    # Each row is a flow, so we return the processed columns directly
    num_data_np = df_processed[numerical_columns_flows].values.astype(np.float32)
    cat_data_np = df_processed[categorical_columns_flows].values.astype(np.int64)
    labels_list = [label] * len(df_processed)  # Assign the file's label to all flows from it

    return file_path, num_data_np, cat_data_np, labels_list


# ──────────────────────────────────────────────────────────────────────────────

def preprocess_flow_vectors_from_csvs(  # Renamed main function
        dataset_dir: str,
        output_file: str,
        test_mode: bool = False,
        rows_per_file: int = 20000,
        # max_seq_len no longer needed
):
    all_files = io_utils.list_csv_files(dataset_dir, recursive=True)  # Assuming recursive
    logging.info(f"[START] Found {len(all_files)} CSV files for flow vector processing.")

    # Checkpoint logic: saves processed data from each file
    # Filename for checkpoint will be like 'somefile_flow_shard.pt'
    pending_files = [f for f in all_files if not (CHECKPOINT_DIR / (Path(f).stem + "_flow_shard.pt")).exists()]
    if len(pending_files) < len(all_files):
        logging.info(
            f"[CHECKPOINT] {len(all_files) - len(pending_files)} files appear to be processed (checkpoints exist).")
    logging.info(f"[CHECKPOINT] {len(pending_files)} files pending processing.")

    # Load DataFrames first (can be memory intensive for many large files)
    loaded_dfs_with_paths = []
    # Using ThreadPoolExecutor for I/O bound task of loading files
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as tpool:
        # io_utils.load_csv_file should return (DataFrame, file_path) or None
        # It should load ALL columns initially, selection happens in process_single_flow_file
        futures = [tpool.submit(io_utils.load_csv_to_df, fp, None, test_mode, rows_per_file) for fp in pending_files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                loaded_dfs_with_paths.append(result)

    if not loaded_dfs_with_paths and not list(CHECKPOINT_DIR.glob("*_flow_shard.pt")):
        raise RuntimeError("No CSV files could be loaded and no flow checkpoints found.")

    # Process DataFrames in parallel
    # process_single_flow_file is CPU bound (standardization, mapping)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ppool:
        # args for process_single_flow_file are (df, file_path)
        futures = [ppool.submit(process_single_flow_file, df_path_tuple) for df_path_tuple in loaded_dfs_with_paths]
        for future in concurrent.futures.as_completed(futures):
            file_path, num_data, cat_data, labels = future.result()
            if num_data is not None and cat_data is not None and labels is not None:
                shard_data_dict = {
                    "numerical_features": torch.from_numpy(num_data),
                    "categorical_features": torch.from_numpy(cat_data),
                    "label": torch.tensor(labels, dtype=torch.long),
                }
                # Save shard to checkpoint
                shard_filename = Path(file_path).stem + "_flow_shard.pt"
                torch.save(shard_data_dict, CHECKPOINT_DIR / shard_filename)
                # logging.info(f"[CKPT] wrote flow shard for {Path(file_path).name}")
            # else: logging.warning(f"Skipped saving shard for {Path(file_path).name} due to processing error or no label.")

    # Aggregate all processed shards from checkpoint directory
    logging.info("Aggregating processed flow shards...")
    all_numerical_tensors: List[torch.Tensor] = []
    all_categorical_tensors: List[torch.Tensor] = []
    all_label_tensors: List[torch.Tensor] = []

    shard_files = sorted(CHECKPOINT_DIR.glob("*_flow_shard.pt"))
    if not shard_files:
        logging.error("No processed flow shards found in checkpoint directory. Cannot create final output.")
        return

    for sf_path in shard_files:
        try:
            data = torch.load(sf_path)
            # Ensure keys exist and tensors are not empty before appending
            if "numerical_features" in data and data["numerical_features"].numel() > 0:
                all_numerical_tensors.append(data["numerical_features"])
            if "categorical_features" in data and data["categorical_features"].numel() > 0:
                all_categorical_tensors.append(data["categorical_features"])
            if "label" in data and data["label"].numel() > 0:
                all_label_tensors.append(data["label"])
        except Exception as e:
            logging.error(f"Error loading shard {sf_path}: {e}")
            continue

    if not all_label_tensors:  # If no labels, means no data was effectively processed
        logging.error("Aggregation resulted in no data. Check processed shards.")
        return

    # Concatenate tensors
    # Handle cases where numerical or categorical features might be absent entirely
    final_numerical_data = torch.cat(all_numerical_tensors, dim=0) if all_numerical_tensors else torch.empty((0, 0),
                                                                                                             dtype=torch.float32)
    final_categorical_data = torch.cat(all_categorical_tensors, dim=0) if all_categorical_tensors else torch.empty(
        (0, 0), dtype=torch.int64)
    final_labels = torch.cat(all_label_tensors, dim=0)

    output_dict = {
        "numerical_features": final_numerical_data,
        "categorical_features": final_categorical_data,
        "label": final_labels,
        "metadata": {  # Add some basic metadata
            "source_dataset_dir": dataset_dir,
            "creation_timestamp": timestamp,
            "numerical_columns_used": numerical_columns_flows,
            "categorical_columns_used": categorical_columns_flows,
        }
    }

    torch.save(output_dict, output_file)
    logging.info(f"✅ [DONE] Merged flow dataset saved to {output_file}")
    logging.info(f"   Total flows processed: {len(final_labels)}")

    label_counts = np.bincount(final_labels.numpy())
    logging.info("[INFO] Final Label distribution:")
    for lbl_id, cnt in enumerate(label_counts):
        label_name = "Unknown_Label"  # Fallback
        for name, val in LABEL_MAPPING.items():
            if val == lbl_id:
                label_name = name
                break
        logging.info(f"  Class '{label_name}' ({lbl_id}): {cnt} flows")


# ──────────────────────────────────────────────────────────────────────────────

def find_label_from_path(file_path: str) -> int:
    # This is your existing function, ensure LABEL_MAPPING is accessible
    current_path = os.path.dirname(file_path)
    while current_path and current_path != os.path.dirname(current_path):
        folder_name = os.path.basename(current_path)
        for key, value in LABEL_MAPPING.items():  # Iterate through items
            if key.lower() in folder_name.lower():
                return value  # Return the mapped integer
        current_path = os.path.dirname(current_path)
    return -1  # No label found


# ──────────────────────────────────────────────────────────────────────────────
# Example Main Execution Block (if you want to run this script directly)
if __name__ == "__main__":
    logging.info("Starting direct execution of FLOW preprocessing script.")

    # --- Configuration for direct run ---
    # **** USER: Adjust these paths and settings for your FLOW data ****
    FLOW_DATA_INPUT_DIR = '../../dataset/roni/DatasetFlow'  # INPUT: Directory with raw FLOW CSVs
    FLOW_DATA_OUTPUT_PT = '../../dataset/processed_flows.pt'  # OUTPUT: Where the final .pt file will be saved

    # This should point to a category_mappings.json file generated for your FLOW features
    # For simplicity, the script assumes a default "category_mappings.json" if not specified.
    # You might want to make this configurable if you have separate mappings for packets and flows.

    TEST_MODE_FLAG = False  # Set to True for quick testing on fewer rows
    ROWS_PER_FILE_LIMIT_IF_TEST = 10000  # Used only if TEST_MODE_FLAG is True

    # --- Ensure necessary files/dirs exist ---
    if not os.path.isdir(FLOW_DATA_INPUT_DIR):
        logging.error(f"❌ Flow data input directory not found: {FLOW_DATA_INPUT_DIR}")
        exit(1)
    # The script will try to load "standardization_stats.npz" and "category_mappings.json"
    # (default path for mappings). Ensure these exist and are appropriate for FLOW data.
    if not os.path.exists(STD_STATS_PATH):
        logging.warning(
            f"🚨 Standardization stats file '{STD_STATS_PATH}' not found. Standardization will fail or use incorrect defaults if columns requiring it are present.")
    if not os.path.exists("category_mappings.json"):  # Default path used by category_mapping.load_mappings()
        logging.warning(
            f"🚨 Default category mappings file 'category_mappings.json' not found. Categorical encoding might fail or be inconsistent.")

    # --- Run the preprocessing ---
    try:
        preprocess_flow_vectors_from_csvs(
            dataset_dir=FLOW_DATA_INPUT_DIR,
            output_file=FLOW_DATA_OUTPUT_PT,
            test_mode=TEST_MODE_FLAG,
            rows_per_file=ROWS_PER_FILE_LIMIT_IF_TEST
        )
    except RuntimeError as e:
        logging.error(f"❌ Runtime Error during flow preprocessing: {e}")
    except FileNotFoundError as e:
        logging.error(f"❌ File Not Found Error during flow preprocessing: {e}")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during flow preprocessing: {type(e).__name__} - {e}",
                      exc_info=True)

    logging.info("🏁 Flow preprocessing script finished. 🏁")
