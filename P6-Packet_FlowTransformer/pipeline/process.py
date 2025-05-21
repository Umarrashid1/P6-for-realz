import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict  # Added Dict
import logging
import datetime
import time  # For timing

from .config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from utils import io_utils
from utils import category_mapping

# ── Set up timestamp and logging ──────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# LOG_FILE is defined globally, but individual scripts might want to control their own logging destination
# For this module, we assume the calling script (e.g., run_pipeline.py) sets up the primary log handler.
# If this script is run standalone or its functions are called where logging isn't configured,
# basicConfig might take over or logs might go to stderr.
# For robust logging, ensure the main execution script configures it.
# The current setup in the user's provided script uses a global LOG_FILE.

# Get a logger specific to this module for clarity in logs
module_logger = logging.getLogger(__name__)  # Changed from 'logging' to 'module_logger' to avoid conflict

# ── Create a timestamped checkpoint directory ──────────────────────────────
# This should ideally be managed by the script calling create_packet_sequences,
# but keeping it here as per the original structure.
CHECKPOINT_DIR_BASE = Path("checkpoints")  # Base directory for checkpoints
# CHECKPOINT_DIR will be set within create_packet_sequences using the global timestamp

# ── Load global numeric stats ───────────────────────────────────────────────
# This assumes STD_STATS_PATH is relative to where this script is executed from,
# or an absolute path.
STD_STATS_PATH = "packet_standardization_stats.npz"
try:
    _std_stats = np.load(STD_STATS_PATH)
    STD_COLS = _std_stats["cols"].tolist()
    GLOBAL_MEAN = dict(zip(STD_COLS, _std_stats["mean"]))
    GLOBAL_STD = dict(zip(STD_COLS, _std_stats["std"]))
    GLOBAL_MEDIAN = dict(zip(STD_COLS, _std_stats["median"]))
    CLIP_LOW = dict(zip(STD_COLS, _std_stats["clip_low"]))
    CLIP_HIGH = dict(zip(STD_COLS, _std_stats["clip_high"]))
    EPS = 1e-6
    module_logger.info(f"Successfully loaded standardization stats from {STD_STATS_PATH}")
except FileNotFoundError:
    module_logger.error(
        f"CRITICAL: Standardization stats file not found at {STD_STATS_PATH}. Preprocessing cannot continue correctly.")
    # Depending on desired behavior, you might raise an error here or allow continuation with warnings.
    # For now, let it proceed, but it will likely fail or produce bad data.
    GLOBAL_MEAN, GLOBAL_STD, GLOBAL_MEDIAN, CLIP_LOW, CLIP_HIGH = {}, {}, {}, {}, {}
    STD_COLS = []
    EPS = 1e-6


# ──────────────────────────────────────────────────────────────────────────────

def process_fragment(args: Tuple[pd.DataFrame, str, int, Path]) -> Tuple[
    str, List[np.ndarray], List[int], List[np.ndarray], Dict[str, List[np.ndarray]]]:
    df, file_path, max_seq_len, current_checkpoint_dir = args  # Added current_checkpoint_dir

    module_logger.info(f"[{Path(file_path).name}] PROCESS_FRAGMENT_START. Initial df rows: {len(df)}")
    frag_time_start = time.time()

    label = find_label_from_path(file_path)
    if label == -1:
        module_logger.warning(f"[{Path(file_path).name}] SKIP: No label found.")
        return file_path, [], [], [], {}

    required_packet_seq_cols = ["stream"]
    # Ensure no duplicates in required_cols
    required_cols_set = set(numerical_columns_packets) | set(categorical_columns_packets) | set(
        required_packet_seq_cols)
    required_cols = list(required_cols_set)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        module_logger.warning(f"[{Path(file_path).name}] SKIP: Missing columns: {missing_cols}")
        return file_path, [], [], [], {}

    # Select only required columns to reduce memory footprint early
    df = df[required_cols].copy()
    module_logger.debug(f"[{Path(file_path).name}] Selected required columns. df shape: {df.shape}")

    # Numerical Feature Preprocessing
    num_proc_start = time.time()
    if not STD_COLS:  # Check if stats were loaded
        module_logger.error(
            f"[{Path(file_path).name}] Standardization stats not loaded. Numerical preprocessing will be incomplete/incorrect.")

    for col in numerical_columns_packets:
        if col in df.columns and col in STD_COLS:  # Ensure column exists and stats exist
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric type
            df[col] = df[col].clip(lower=CLIP_LOW[col], upper=CLIP_HIGH[col])
            df[col] = df[col].fillna(GLOBAL_MEDIAN[col])
            df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)
        elif col not in STD_COLS and col in df.columns:
            module_logger.warning(
                f"[{Path(file_path).name}] No standardization stats for numerical column '{col}'. Skipping its standardization, filling NaNs with 0.")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # If col not in df.columns, it's handled by missing_cols check earlier
    module_logger.info(
        f"[{Path(file_path).name}] Numerical processing completed in {time.time() - num_proc_start:.2f}s.")

    # Categorical Feature Preprocessing
    cat_proc_start = time.time()
    try:
        cat_mappings = category_mapping.load_mappings(is_flow=False)  # keys are original types
    except Exception as e:
        module_logger.error(
            f"[{Path(file_path).name}] CRITICAL: Failed to load category_mappings_packets.json: {e}. Skipping categorical processing for this file.")
        # Fallback: create empty categorical features or error out
        # For now, let's try to proceed by making them all 'unknown_id' equivalent (e.g., 0)
        # This is a severe issue, so the output quality will be affected.
        for col in categorical_columns_packets:
            df[col] = 0  # Assign a default integer ID
        cat_mappings = {}  # Empty dict to avoid errors later if accessed

    for col in categorical_columns_packets:
        col_map_time_start = time.time()
        if col not in df.columns:
            module_logger.warning(
                f"[{Path(file_path).name}] Categorical column {col} unexpectedly missing after initial selection. Filling with 0.")
            df[col] = 0
            continue

        mapping = cat_mappings.get(col)
        if mapping is None:
            module_logger.warning(
                f"[{Path(file_path).name}] No mapping found for categorical column {col} in loaded mappings. Filling with 0.")
            df[col] = 0
            continue

        # "unknown" (string) is a guaranteed key in your mappings, its value is the integer ID for unknowns
        # This relies on category_mapping.py correctly creating this entry.
        unknown_id = mapping.get("unknown")
        if unknown_id is None:
            module_logger.warning(
                f"[{Path(file_path).name}] 'unknown' key not found in mapping for {col}. Using 0 as unknown_id.")
            unknown_id = 0  # Fallback unknown_id

        # Convert column to string for .map if mapping keys are strings,
        # OR ensure types match if mapping keys are original types.
        # Given category_mapping.py uses ast.literal_eval, keys in `mapping` are original types.
        # The most robust way is to ensure df[col] has compatible types or handle type errors.
        # For simplicity and to avoid errors if CSVs have mixed types that pandas reads as 'object':
        # We will attempt to map. If it fails due to type, we might need more specific type handling.

        # NaNs in the original df[col] will become NaN after .map if not in mapping keys.
        # Values in df[col] not present in mapping.keys() will also become NaN after .map.
        # We then fill all these NaNs with unknown_id.
        try:
            # Ensure the column is of a type that can be reasonably mapped.
            # If a column is purely numeric but represents categories (e.g. port numbers),
            # and mapping keys are integers, this should work.
            # If a column is string/object, and mapping keys are strings, this should work.
            # The critical part is that the *values* in df[col] match the *type of keys* in mapping.

            # Handle NaNs explicitly first. If we fillna with a string like "unknown_placeholder_for_nan",
            # then "unknown_placeholder_for_nan" must be a key in the mapping.
            # Simpler: map directly, then fill NaNs created by map (key not found) or original NaNs.

            mapped_series = df[col].map(mapping)
            # Fill NaNs that resulted from .map() (value not in mapping or original NaN) with unknown_id
            df[col] = mapped_series.fillna(unknown_id).astype(int)

        except TypeError as te:
            module_logger.error(
                f"[{Path(file_path).name}] TypeError during .map() for column {col}: {te}. Trying .apply() as fallback.")
            # Fallback to .apply() if .map() fails due to unhashable types or other complex type issues
            # This was the user's original working version.
            df[col] = df[col].apply(lambda x: mapping.get(x, unknown_id)).astype(int)
        except Exception as e:
            module_logger.error(f"[{Path(file_path).name}] Error mapping column {col}: {e}. Filling with unknown_id.")
            df[col] = unknown_id  # Fill entire column with unknown_id on other errors

        log_level = logging.DEBUG if len(df[col]) < 10000 else logging.INFO  # Avoid verbose logs for huge columns
        module_logger.log(log_level,
                          f"[{Path(file_path).name}] Mapped column {col} in {time.time() - col_map_time_start:.4f}s. Unique values after map: {df[col].nunique()}")

    module_logger.info(
        f"[{Path(file_path).name}] Categorical processing completed in {time.time() - cat_proc_start:.2f}s.")

    # Grouping and Sequence Generation
    group_time_start = time.time()
    try:
        packet_groups = df.groupby("stream", sort=False, observed=True)  # observed=True can be faster
    except Exception as e:
        module_logger.error(
            f"[{Path(file_path).name}] Error during groupby('stream'): {e}. Skipping sequence generation.")
        return file_path, [], [], [], {}

    num_groups = len(packet_groups)  # More efficient way to get group count
    module_logger.info(
        f"[{Path(file_path).name}] Grouped by 'stream' in {time.time() - group_time_start:.2f}s. Number of groups: {num_groups}")

    seq_gen_time_start = time.time()
    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    # Initialize cat_arrays with empty lists for all categorical_columns_packets
    cat_arrays: Dict[str, List[np.ndarray]] = {col: [] for col in categorical_columns_packets}

    sequences_generated_count = 0

    for stream_id, packet_seq_df in packet_groups:
        if packet_seq_df.empty:
            continue

        # Log only a sample of stream processing if too many streams
        if sequences_generated_count < 5 or sequences_generated_count % 1000 == 0:  # Log first 5 then every 1000th
            module_logger.debug(
                f"[{Path(file_path).name}] Processing stream_id: {stream_id}, length: {len(packet_seq_df)}")

        num_feat = packet_seq_df[numerical_columns_packets].values

        # Ensure cat_feat dictionary is correctly populated
        cat_feat_for_stream = {}
        for col in categorical_columns_packets:
            if col in packet_seq_df:
                cat_feat_for_stream[col] = packet_seq_df[col].values
            else:  # Should not happen if columns are selected correctly earlier
                module_logger.warning(
                    f"[{Path(file_path).name}] Stream {stream_id}: Categorical column {col} missing. Filling with zeros.")
                cat_feat_for_stream[col] = np.zeros(len(packet_seq_df), dtype=np.int64)  # Default to zeros

        packet_seq_len = len(num_feat)

        def pad_and_append(start_idx, end_idx):
            nonlocal sequences_generated_count
            num_slice = num_feat[start_idx:end_idx]
            pkt_arrays.append(num_slice.astype(np.float32))

            current_slice_len = len(num_slice)
            mask = np.ones(max_seq_len, dtype=np.float32)
            if current_slice_len < max_seq_len:  # Should only happen if original seq_len < max_seq_len and this is the only slice
                mask[current_slice_len:] = 0
            mask_arrays.append(mask)

            label_list.append(label)

            for col_name in categorical_columns_packets:
                # unknown_id_for_padding = cat_mappings.get(col_name, {}).get("unknown", 0) # Get unknown_id for this specific column
                # Default to 0 if mapping or "unknown" key is missing, though it should exist from earlier logic
                col_mapping = cat_mappings.get(col_name)
                if col_mapping:
                    unknown_id_for_padding = col_mapping.get("unknown")
                    if unknown_id_for_padding is None:  # Should not happen if cat_mappings loaded correctly
                        unknown_id_for_padding = 0
                else:  # Should not happen if cat_mappings loaded correctly
                    unknown_id_for_padding = 0

                cat_slice = cat_feat_for_stream[col_name][start_idx:end_idx]
                padded_cat_slice = np.pad(cat_slice, (0, max_seq_len - len(cat_slice)),
                                          mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays[col_name].append(padded_cat_slice.astype(np.int64))
            sequences_generated_count += 1

        if packet_seq_len == 0:  # Should be caught by packet_seq_df.empty earlier
            continue

        if packet_seq_len < max_seq_len:
            pad_len = max_seq_len - packet_seq_len
            padded_num_feat = np.concatenate(
                [num_feat, np.zeros((pad_len, len(numerical_columns_packets)), dtype=np.float32)], axis=0)
            pkt_arrays.append(padded_num_feat)

            mask = np.concatenate([np.ones(packet_seq_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
            mask_arrays.append(mask)
            label_list.append(label)

            for col_name in categorical_columns_packets:
                col_mapping = cat_mappings.get(col_name)
                if col_mapping:
                    unknown_id_for_padding = col_mapping.get("unknown")
                    if unknown_id_for_padding is None: unknown_id_for_padding = 0
                else:
                    unknown_id_for_padding = 0

                padded_cat_col = np.pad(cat_feat_for_stream[col_name], (0, pad_len),
                                        mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays[col_name].append(padded_cat_col.astype(np.int64))
            sequences_generated_count += 1
        else:
            stride = max_seq_len // 2
            if stride == 0: stride = 1  # Avoid stride 0 for very small max_seq_len

            for start in range(0, packet_seq_len - max_seq_len + 1, stride):
                pad_and_append(start, start + max_seq_len)

            # Handle the tail end if not perfectly divisible by stride
            if (packet_seq_len - max_seq_len) % stride != 0 and packet_seq_len > max_seq_len:
                # This condition ensures we only pad the very last segment if it's shorter than max_seq_len
                # The start index should be such that it captures the last max_seq_len elements
                start_of_last_segment = packet_seq_len - max_seq_len
                pad_and_append(start_of_last_segment, packet_seq_len)

    module_logger.info(
        f"[{Path(file_path).name}] Sequence generation loop completed in {time.time() - seq_gen_time_start:.2f}s. Generated {sequences_generated_count} sequences.")
    module_logger.info(
        f"[{Path(file_path).name}] PROCESS_FRAGMENT_END. Total time: {time.time() - frag_time_start:.2f}s. Output sequences: {len(pkt_arrays)}")
    return file_path, pkt_arrays, label_list, mask_arrays, cat_arrays


# ──────────────────────────────────────────────────────────────────────────────

def create_packet_sequences(
        dataset_dir: str,
        output_file: str,
        test_mode: bool = False,
        rows_per_file: int = 0,  # Default to 0 (all rows) if not in test_mode
        max_seq_len: int = 64,
):
    # Ensure global timestamp is used for this run's checkpoint directory
    # This needs to be coordinated if create_packet_sequences is called multiple times
    # or if the script is re-run. For a single execution, this is fine.
    global timestamp  # Use the global timestamp defined at the start of the script
    current_checkpoint_dir = CHECKPOINT_DIR_BASE / timestamp
    current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    module_logger.info(f"Using checkpoint directory for this run: {current_checkpoint_dir}")

    module_logger.info(f"--- Starting create_packet_sequences ---")
    module_logger.info(f"Dataset dir: {dataset_dir}, Output file: {output_file}")
    module_logger.info(f"Test mode: {test_mode}, Rows per file: {rows_per_file}, Max seq len: {max_seq_len}")

    # Adjust rows_per_file if not in test_mode and rows_per_file is 0 (meaning all rows)
    effective_rows_per_file = rows_per_file
    if not test_mode and rows_per_file == 0:
        effective_rows_per_file = None  # pandas read_csv reads all rows if nrows=None
        module_logger.info("Processing all rows per file (test_mode=False, rows_per_file=0).")
    elif test_mode and rows_per_file == 0:
        module_logger.warning(
            "Test mode is True but rows_per_file is 0. This will read all rows. Consider setting rows_per_file > 0 for test mode.")
        effective_rows_per_file = None

    all_files = io_utils.list_csv_files(dataset_dir)
    module_logger.info(f"[START] Found {len(all_files)} CSV files.")

    pending_files = [f for f in all_files if not (current_checkpoint_dir / (Path(f).stem + ".pt")).exists()]
    if len(pending_files) < len(all_files):
        module_logger.info(
            f"[CHECKPOINT] {len(all_files) - len(pending_files)} files appear to be processed from previous run in this session (checkpoints exist in {current_checkpoint_dir}).")
    module_logger.info(f"[CHECKPOINT] {len(pending_files)} files pending processing for this run.")

    if not pending_files:
        module_logger.info("No files pending processing. Aggregating existing shards if any.")
    else:
        # --- ThreadPool for loading, ProcessPool for processing ---
        # This pattern loads all DFs first, then processes. Can be memory intensive.
        # For very large datasets, a more streaming approach (load one, process one) might be better.
        loading_pool_start_time = time.time()
        loaded_dfs_with_paths = []
        # Max workers for ThreadPool can be higher for I/O, os.cpu_count() is a safe default.
        # Python's default for ThreadPoolExecutor is min(32, os.cpu_count() + 4) in newer versions.
        # Let's use a slightly higher number for I/O if many small files, but os.cpu_count() is fine.
        num_load_workers = os.cpu_count() * 2 if os.cpu_count() else 4  # Example: up to 2x CPU cores for I/O

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
            futures_load = {
                tpool.submit(io_utils.load_csv_file, fp, test_mode, effective_rows_per_file): Path(fp).name
                for fp in pending_files
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures_load)):
                file_name = futures_load[future]
                try:
                    result = future.result()
                    if result is not None:
                        loaded_dfs_with_paths.append(result)
                        module_logger.info(
                            f"Loaded file {i + 1}/{len(pending_files)}: {file_name} (Shape: {result[0].shape if result[0] is not None else 'N/A'})")
                    else:
                        module_logger.warning(f"Skipped file (None result from load): {file_name}")
                except Exception as e:
                    module_logger.error(f"Error loading file {file_name}: {e}",
                                        exc_info=False)  # Set exc_info=True for full traceback
        module_logger.info(
            f"Finished loading {len(loaded_dfs_with_paths)} DataFrames in {time.time() - loading_pool_start_time:.2f}s.")

        if not loaded_dfs_with_paths and not list(current_checkpoint_dir.glob("*.pt")):
            # Check if any shards exist from a previous incomplete run within this timestamped folder
            module_logger.error(
                "No CSV files could be loaded and no checkpoints found in the current session's checkpoint directory. Exiting.")
            raise RuntimeError("No files loaded and no checkpoints found for this run.")

        # Prepare arguments for process_fragment, ensuring df is not None
        proc_args = [(df, path, max_seq_len, current_checkpoint_dir) for df, path in loaded_dfs_with_paths if
                     df is not None]
        if not proc_args:
            module_logger.warning("No valid DataFrames to process after loading stage.")
        else:
            module_logger.info(f"Submitting {len(proc_args)} loaded DataFrames for parallel processing...")
            processing_pool_start_time = time.time()
            # For CPU-bound tasks, os.cpu_count() is appropriate for ProcessPoolExecutor
            num_process_workers = os.cpu_count() if os.cpu_count() else 1
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
                # Using map for potentially better memory management with iterators, but future.result() gives more control for logging
                results_iter = ppool.map(process_fragment, proc_args,
                                         chunksize=1)  # chunksize=1 for better progress feedback

                processed_count = 0
                for file_path_processed, pkt_arrs, lbls, masks, cat_arrs in results_iter:
                    processed_count += 1
                    module_logger.info(
                        f"Processing result {processed_count}/{len(proc_args)} for: {Path(file_path_processed).name}")
                    if not pkt_arrs:  # If process_fragment returned empty lists (e.g., skipped file)
                        module_logger.warning(
                            f"No sequences generated for {Path(file_path_processed).name}, skipping shard save.")
                        continue

                    try:
                        shard_dict_start_time = time.time()
                        shard = {
                            "packet_seq": torch.from_numpy(np.stack(pkt_arrs)).float(),  # Ensure float32
                            "label": torch.tensor(lbls, dtype=torch.long),
                            "attention_mask": torch.from_numpy(np.stack(masks)).float(),  # Ensure float32
                        }
                        for col_name_cat in categorical_columns_packets:  # Use the definitive list
                            if col_name_cat in cat_arrs and cat_arrs[col_name_cat]:  # Check if list is not empty
                                shard[col_name_cat] = torch.from_numpy(
                                    np.stack(cat_arrs[col_name_cat])).long()  # Ensure int64
                            else:  # Handle case where a categorical column might have no data (e.g. all skipped)
                                module_logger.warning(
                                    f"No data for categorical column '{col_name_cat}' in shard for {Path(file_path_processed).name}. Creating empty tensor.")
                                shard[col_name_cat] = torch.empty((len(pkt_arrs), max_seq_len),
                                                                  dtype=torch.long)  # Match shape N,T

                        module_logger.debug(
                            f"Shard dictionary for {Path(file_path_processed).name} created in {time.time() - shard_dict_start_time:.4f}s")

                        save_shard_start_time = time.time()
                        torch.save(shard, current_checkpoint_dir / (Path(file_path_processed).stem + ".pt"))
                        module_logger.info(
                            f"[CKPT] wrote shard for {Path(file_path_processed).name} in {time.time() - save_shard_start_time:.2f}s")
                    except Exception as e:
                        module_logger.error(f"Error creating or saving shard for {Path(file_path_processed).name}: {e}",
                                            exc_info=True)

            module_logger.info(
                f"Finished processing all submitted DataFrames in {time.time() - processing_pool_start_time:.2f}s.")

    # --- Aggregation ---
    module_logger.info(f"Starting aggregation of shards from: {current_checkpoint_dir}")
    aggregation_start_time = time.time()

    pkt_tensors, lbl_tensors, msk_tensors = [], [], []
    # Initialize cat_tensors dict with all categorical columns to ensure all keys exist
    cat_tensors: Dict[str, List[torch.Tensor]] = {col: [] for col in categorical_columns_packets}

    # It's crucial that current_checkpoint_dir points to the correct directory for this run
    shard_files = sorted(list(current_checkpoint_dir.glob("*.pt")))  # Ensure it's a list for len()
    module_logger.info(f"Found {len(shard_files)} shard files for aggregation.")

    if not shard_files:
        module_logger.error(
            f"No shard files found in {current_checkpoint_dir} to aggregate. Cannot create output file: {output_file}")
        return  # Exit if no shards

    for i, sf_path in enumerate(shard_files):
        module_logger.debug(f"Loading shard {i + 1}/{len(shard_files)}: {sf_path.name}")
        try:
            data = torch.load(sf_path)  # Load to CPU by default if not specified, which is fine for aggregation
            if "packet_seq" in data and data["packet_seq"].numel() > 0:  # Check if tensor is not empty
                pkt_tensors.append(data["packet_seq"])
                lbl_tensors.append(data["label"])
                msk_tensors.append(data["attention_mask"])
                for col in categorical_columns_packets:
                    if col in data and data[col].numel() > 0:
                        cat_tensors[col].append(data[col])
                    elif col in data and data[
                        col].ndim == 2:  # Allow empty tensors if they have correct ndim (N, T_actual)
                        cat_tensors[col].append(data[col])
                    else:  # If key missing or malformed, create placeholder to avoid cat error
                        module_logger.warning(
                            f"Categorical column '{col}' missing or malformed in shard {sf_path.name}. Appending placeholder.")
                        # Create a placeholder of shape [num_sequences_in_this_shard, max_seq_len]
                        # This assumes 'packet_seq' exists and is valid for this shard
                        num_seq_in_shard = data["packet_seq"].shape[0] if "packet_seq" in data else 0
                        cat_tensors[col].append(torch.zeros((num_seq_in_shard, max_seq_len), dtype=torch.long))
            else:
                module_logger.warning(f"Skipping shard {sf_path.name} as 'packet_seq' is missing or empty.")
        except Exception as e:
            module_logger.error(f"Error loading or processing shard {sf_path.name} for aggregation: {e}", exc_info=True)
            continue  # Skip problematic shard

    if not pkt_tensors:  # If all shards were problematic or empty
        module_logger.error(
            f"No valid data found in any shards from {current_checkpoint_dir}. Cannot create output file: {output_file}")
        return

    final_output_dict: Dict[str, torch.Tensor] = {}
    try:
        final_output_dict["packet_seq"] = torch.cat(pkt_tensors, dim=0)
        final_output_dict["label"] = torch.cat(lbl_tensors, dim=0)
        final_output_dict["attention_mask"] = torch.cat(msk_tensors, dim=0)

        for col in categorical_columns_packets:
            if cat_tensors[col]:  # If list is not empty
                final_output_dict[col] = torch.cat(cat_tensors[col], dim=0)
            else:  # If all shards missed this cat feature or had empty lists for it
                module_logger.warning(
                    f"No data aggregated for categorical column '{col}'. Creating empty tensor in final output.")
                # Create an empty tensor with 0 sequences but correct second dimension (max_seq_len)
                # The number of sequences (dim 0) will be consistent with packet_seq due to torch.cat later
                # This tensor will have shape [Total_Sequences, max_seq_len] if other tensors are valid
                # Or [0, max_seq_len] if pkt_tensors was also empty (caught earlier)
                num_total_sequences = final_output_dict["packet_seq"].shape[
                    0] if "packet_seq" in final_output_dict else 0
                final_output_dict[col] = torch.empty((num_total_sequences, max_seq_len), dtype=torch.long)

    except Exception as e:
        module_logger.error(f"Error during final tensor concatenation: {e}", exc_info=True)
        return

    module_logger.info(f"Aggregation completed in {time.time() - aggregation_start_time:.2f}s.")

    save_output_start_time = time.time()
    torch.save(final_output_dict, output_file)
    module_logger.info(f"[DONE] Merged dataset saved to {output_file} in {time.time() - save_output_start_time:.2f}s")

    try:
        if "label" in final_output_dict and final_output_dict["label"].numel() > 0:
            label_counts = np.bincount(final_output_dict["label"].cpu().numpy())  # Move to CPU for numpy
            module_logger.info("Final Label distribution in merged dataset:")
            for lbl_id, cnt in enumerate(label_counts):
                if cnt > 0:  # Only print classes that are present
                    # Find label name from LABEL_MAPPING
                    label_name = "Unknown"
                    for name, val in LABEL_MAPPING.items():
                        if val == lbl_id:
                            label_name = name
                            break
                    module_logger.info(f"  Class '{label_name}' ({lbl_id}): {cnt} sequences")
        else:
            module_logger.warning("No 'label' data in final output to report distribution.")
    except Exception as e:
        module_logger.error(f"Error reporting label distribution: {e}")

    module_logger.info(f"--- create_packet_sequences Finished ---")


# ──────────────────────────────────────────────────────────────────────────────

def find_label_from_path(file_path: str) -> int:
    # This function seems robust.
    current_path = Path(file_path).parent  # Use pathlib for robustness
    while current_path != current_path.parent:  # Loop until root or no change
        folder_name = current_path.name
        for key, value in LABEL_MAPPING.items():  # Iterate through items
            if key.lower() in folder_name.lower():
                return value
        current_path = current_path.parent
    return -1  # No label found
