import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import datetime
import time
import random

from .config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from utils import io_utils
from utils import category_mapping

# ── Set up logging ─────────────────
LOG_FILE_BASENAME = "preprocessing_packets"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{LOG_FILE_BASENAME}_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

# ── FIXED Checkpoint directory ─────────────────────────
CHECKPOINT_DIR = Path("checkpoints_packet_shards")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Using FIXED checkpoint directory for this run: {CHECKPOINT_DIR.resolve()}")

# ── Load global numeric stats ─────────────────
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
    logging.info(f"Successfully loaded standardization stats from {STD_STATS_PATH}")
except FileNotFoundError:
    logging.error(
        f"CRITICAL: Standardization stats file not found at {STD_STATS_PATH}. Preprocessing cannot continue correctly.")
    raise SystemExit(f"CRITICAL: Standardization stats file not found at {STD_STATS_PATH}")


def process_fragment(args: Tuple[pd.DataFrame, str, int]) -> Tuple[
    str, Optional[List[np.ndarray]], Optional[List[int]], Optional[List[np.ndarray]], Optional[
        Dict[str, List[np.ndarray]]]]:
    df, file_path, max_seq_len = args

    logging.info(f"[{Path(file_path).name}] PROCESS_FRAGMENT_START. Initial df rows: {len(df)}")
    frag_time_start = time.time()

    label = find_label_from_path(file_path)
    if label == -1:
        logging.warning(f"[{Path(file_path).name}] SKIP: No label found.")
        return file_path, None, None, None, None

    required_packet_seq_cols = ["stream"]
    required_cols_set = set(numerical_columns_packets) | set(categorical_columns_packets) | set(
        required_packet_seq_cols)
    required_cols = list(required_cols_set)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logging.warning(f"[{Path(file_path).name}] SKIP: Missing columns: {missing_cols}")
        return file_path, None, None, None, None

    df = df[required_cols].copy()
    logging.debug(f"[{Path(file_path).name}] Selected required columns. df shape: {df.shape}")

    num_proc_start = time.time()
    for col_num_proc in numerical_columns_packets:
        if col_num_proc in df.columns and col_num_proc in STD_COLS:
            df[col_num_proc] = pd.to_numeric(df[col_num_proc], errors='coerce')
            df[col_num_proc] = df[col_num_proc].clip(lower=CLIP_LOW[col_num_proc], upper=CLIP_HIGH[col_num_proc])
            df[col_num_proc] = df[col_num_proc].fillna(GLOBAL_MEDIAN[col_num_proc])
            df[col_num_proc] = (df[col_num_proc] - GLOBAL_MEAN[col_num_proc]) / (GLOBAL_STD[col_num_proc] + EPS)
        elif col_num_proc not in STD_COLS and col_num_proc in df.columns:
            logging.warning(
                f"[{Path(file_path).name}] No standardization stats for numerical column '{col_num_proc}'. Skipping its standardization, filling NaNs with 0.")
            df[col_num_proc] = pd.to_numeric(df[col_num_proc], errors='coerce').fillna(0)
        elif col_num_proc not in df.columns:
            logging.error(
                f"[{Path(file_path).name}] Numerical column '{col_num_proc}' configured but not found. Filling with 0.")
            df[col_num_proc] = 0
    logging.info(f"[{Path(file_path).name}] Numerical processing completed in {time.time() - num_proc_start:.2f}s.")

    cat_proc_start = time.time()
    try:
        cat_mappings = category_mapping.load_mappings(is_flow=False)
    except Exception as e:
        logging.error(
            f"[{Path(file_path).name}] CRITICAL: Failed to load category_mappings_packets.json: {e}. Assigning default 0 to categorical columns.")
        for col_cat_proc_init_err in categorical_columns_packets:  # Unique name for loop var
            if col_cat_proc_init_err in df.columns: df[col_cat_proc_init_err] = 0
        cat_mappings = {}

    for col_cat_proc in categorical_columns_packets:
        col_map_time_start = time.time()
        if col_cat_proc not in df.columns:
            logging.warning(
                f"[{Path(file_path).name}] Categorical column {col_cat_proc} unexpectedly missing. Filling with 0.")
            df[col_cat_proc] = 0
            continue

        df[col_cat_proc] = df[col_cat_proc].astype(str).fillna("unknown")
        mapping = cat_mappings.get(col_cat_proc)
        unknown_id = 0
        if mapping:
            # Ensure "unknown" key is correctly retrieved (repr vs str)
            # Assuming load_mappings stores keys as original values (str, int, float)
            # and "unknown" is stored as the string "unknown"
            unknown_id = mapping.get("unknown")  # Try direct string "unknown"
            if unknown_id is None:  # Fallback if it was stored as repr("unknown") or other forms
                unknown_id = mapping.get(repr("unknown"), 0)  # Default to 0 if not found
        else:
            logging.warning(
                f"[{Path(file_path).name}] No mapping found for categorical column {col_cat_proc}. Using 0 as unknown_id for all values.")

        temp_mapped_values = []
        if mapping:
            for val_in_df in df[col_cat_proc]:
                # Try direct match (original value type)
                mapped_val = mapping.get(val_in_df)
                if mapped_val is None:
                    # Try as string if original might have been numeric but mapping keys are strings
                    mapped_val = mapping.get(str(val_in_df))
                if mapped_val is None:
                    # Try common literal representations if original was string but mapping keys might be literals
                    try:
                        mapped_val = mapping.get(int(val_in_df))
                    except (ValueError, TypeError):
                        pass  # Ignore if not int-like
                if mapped_val is None:
                    try:
                        mapped_val = mapping.get(float(val_in_df))
                    except (ValueError, TypeError):
                        pass  # Ignore if not float-like

                temp_mapped_values.append(mapped_val if mapped_val is not None else unknown_id)
            df[col_cat_proc] = pd.Series(temp_mapped_values, index=df.index).astype(int)
        else:
            df[col_cat_proc] = unknown_id

        logging.debug(
            f"[{Path(file_path).name}] Mapped column {col_cat_proc} in {time.time() - col_map_time_start:.4f}s.")

    logging.info(f"[{Path(file_path).name}] Categorical processing completed in {time.time() - cat_proc_start:.2f}s.")

    if 'stream' not in df.columns:
        logging.error(f"[{Path(file_path).name}] 'stream' column missing for groupby. Cannot proceed.")
        return file_path, None, None, None, None

    # Convert 'stream' to a consistent type before groupby, e.g., string, to avoid issues with mixed types.
    # This can prevent `observed=True` from behaving unexpectedly if 'stream' has numeric-like strings and actual numbers.
    df['stream'] = df['stream'].astype(str)
    packet_groups = df.groupby("stream", sort=False, observed=False)  # Set observed=False if 'stream' is now string

    num_groups = len(packet_groups) if hasattr(packet_groups, '__len__') else packet_groups.ngroups
    logging.info(
        f"[{Path(file_path).name}] Grouped by 'stream'. Number of groups: {num_groups}")

    seq_gen_time_start = time.time()
    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    cat_arrays_dict_for_fragment: Dict[str, List[np.ndarray]] = {c: [] for c in categorical_columns_packets}
    sequences_generated_count = 0

    for stream_id, packet_seq_df in packet_groups:
        if packet_seq_df.empty: continue

        # Ensure all numerical columns exist, fill with 0 if missing from this specific group
        num_feat_df_group = packet_seq_df.reindex(columns=numerical_columns_packets, fill_value=0)
        num_feat = num_feat_df_group[numerical_columns_packets].values

        cat_feat_for_stream = {}
        for c_col_seq_loop in categorical_columns_packets:  # Unique name for loop var
            if c_col_seq_loop in packet_seq_df:
                cat_feat_for_stream[c_col_seq_loop] = packet_seq_df[c_col_seq_loop].values
            else:  # Should be handled by reindex for categorical if we did that, or ensure 0s
                cat_feat_for_stream[c_col_seq_loop] = np.zeros(len(num_feat), dtype=np.int64)

        packet_seq_len = len(num_feat)
        if packet_seq_len == 0: continue

        def pad_and_append_segment(start_idx, end_idx):
            nonlocal sequences_generated_count
            num_slice = num_feat[start_idx:end_idx]

            # Ensure num_slice has 2 dimensions for padding
            if num_slice.ndim == 1:  # Should not happen if numerical_columns_packets is not empty
                num_slice = num_slice.reshape(-1, 1) if len(numerical_columns_packets) == 1 else num_slice.reshape(-1,
                                                                                                                   len(numerical_columns_packets))

            padded_num_slice = np.pad(num_slice, ((0, max_seq_len - len(num_slice)), (0, 0)), mode='constant',
                                      constant_values=0)
            pkt_arrays.append(padded_num_slice.astype(np.float32))

            current_slice_len = len(num_slice)
            mask = np.ones(max_seq_len, dtype=np.float32)
            if current_slice_len < max_seq_len: mask[current_slice_len:] = 0
            mask_arrays.append(mask)
            label_list.append(label)

            for col_name_cat_seq_pad in categorical_columns_packets:  # Unique name for loop var
                # Default to 0 if mapping or "unknown" is not found for padding
                current_col_mapping = cat_mappings.get(col_name_cat_seq_pad, {})
                unknown_id_for_padding = current_col_mapping.get("unknown", 0)

                cat_original_slice = cat_feat_for_stream.get(col_name_cat_seq_pad, np.array([], dtype=np.int64))[
                                     start_idx:end_idx]

                if cat_original_slice.ndim > 1: cat_original_slice = cat_original_slice.flatten()

                # Ensure cat_original_slice is 1D for padding
                if not isinstance(cat_original_slice, np.ndarray):  # Convert if it's a list or other
                    cat_original_slice = np.array(cat_original_slice, dtype=np.int64)
                if cat_original_slice.ndim == 0 and cat_original_slice.size == 1:  # Handle scalar case
                    cat_original_slice = np.array([cat_original_slice.item()], dtype=np.int64)
                elif cat_original_slice.ndim == 0 and cat_original_slice.size == 0:  # Handle empty scalar
                    cat_original_slice = np.array([], dtype=np.int64)

                padded_cat_slice = np.pad(cat_original_slice, (0, max_seq_len - len(cat_original_slice)),
                                          mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays_dict_for_fragment[col_name_cat_seq_pad].append(padded_cat_slice.astype(np.int64))
            sequences_generated_count += 1

        if packet_seq_len <= max_seq_len:
            pad_and_append_segment(0, packet_seq_len)
        else:
            stride = max_seq_len // 2
            if stride == 0: stride = 1  # Avoid infinite loop if max_seq_len is 1
            for start in range(0, packet_seq_len - max_seq_len + 1, stride):
                pad_and_append_segment(start, start + max_seq_len)

            # Always include the very last segment if it wasn't covered by striding
            # This ensures the tail end of long sequences is always processed.
            last_segment_start_index = packet_seq_len - max_seq_len
            if last_segment_start_index > 0:  # only if sequence is longer than max_seq_len
                # Check if this last segment is different from the one generated by the last stride step
                last_stride_generated_start = ((packet_seq_len - max_seq_len) // stride) * stride if stride > 0 else 0
                if last_segment_start_index > last_stride_generated_start:
                    pad_and_append_segment(last_segment_start_index, packet_seq_len)

    logging.info(
        f"[{Path(file_path).name}] Sequence generation loop completed in {time.time() - seq_gen_time_start:.2f}s. Generated {sequences_generated_count} sequences.")
    if not pkt_arrays:
        logging.warning(f"[{Path(file_path).name}] No sequences generated. Returning empty lists.")
        return file_path, [], [], [], {c: [] for c in categorical_columns_packets}

    logging.info(
        f"[{Path(file_path).name}] PROCESS_FRAGMENT_END. Total time: {time.time() - frag_time_start:.2f}s. Output sequences: {len(pkt_arrays)}")
    return file_path, pkt_arrays, label_list, mask_arrays, cat_arrays_dict_for_fragment


def create_packet_sequences(
        dataset_dir: str,
        output_file: str,
        test_mode: bool = False,
        rows_per_file: int = 0,
        max_seq_len: int = 64,
        files_per_processing_batch: int = 30
):
    logging.info(f"--- Starting create_packet_sequences (Packet Data) ---")
    logging.info(f"Dataset dir: {dataset_dir}, Output file: {output_file}")
    logging.info(f"Test mode: {test_mode}, Rows per file (0 for all): {rows_per_file}, Max seq len: {max_seq_len}")
    logging.info(f"Files per internal processing batch: {files_per_processing_batch}")

    effective_rows_per_file = rows_per_file if rows_per_file > 0 else None
    if not test_mode and rows_per_file == 0:
        logging.info("Processing all rows per file.")
    elif test_mode and rows_per_file == 0:
        logging.warning("Test mode is True but rows_per_file is 0. Processing all rows.")

    all_files_full_list = sorted(io_utils.list_csv_files(dataset_dir))
    logging.info(f"Found {len(all_files_full_list)} CSV files in dataset directory.")

    if not all_files_full_list:
        logging.error("No CSV files found. Exiting.")
        raise RuntimeError("No CSV files found in dataset directory.")

    # The main loop now focuses on ensuring shards are created.
    # In-memory accumulation lists (final_pkt_tensors, etc.) are removed from this stage.
    processed_shards_count = 0

    for i in range(0, len(all_files_full_list), files_per_processing_batch):
        current_batch_file_paths = all_files_full_list[i:i + files_per_processing_batch]
        batch_number = (i // files_per_processing_batch) + 1
        total_batches = (len(all_files_full_list) + files_per_processing_batch - 1) // files_per_processing_batch

        logging.info(
            f"\n--- Processing Batch {batch_number}/{total_batches} ({len(current_batch_file_paths)} files) to ensure shards exist ---")

        pending_files_in_batch_paths = []
        for fp_in_batch in current_batch_file_paths:
            shard_path = CHECKPOINT_DIR / (Path(fp_in_batch).stem + ".pt")
            if shard_path.exists():
                logging.debug(
                    f"[BATCH {batch_number}] Checkpoint for {Path(fp_in_batch).name} already exists. Skipping processing for this file.")
                processed_shards_count += 1  # Count existing shards as processed for this run's purpose
                continue
            pending_files_in_batch_paths.append(fp_in_batch)

        if not pending_files_in_batch_paths:
            logging.info(f"[BATCH {batch_number}] All files in this batch already have shards. Moving to next batch.")
            continue

        logging.info(
            f"[BATCH {batch_number}] {len(pending_files_in_batch_paths)} files in this batch require shard creation.")

        loading_pool_start_time = time.time()
        loaded_dfs_for_batch = []
        # Consider reducing num_load_workers if memory is an issue during loading itself
        num_load_workers = 12
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
            futures_load_batch = {
                tpool.submit(io_utils.load_csv_file, fp_load_b, test_mode, effective_rows_per_file): Path(
                    fp_load_b).name
                for fp_load_b in pending_files_in_batch_paths
            }
            for i_load_b, future_load_b in enumerate(concurrent.futures.as_completed(futures_load_batch)):
                file_name_loaded_b = futures_load_batch[future_load_b]
                try:
                    result_load_b = future_load_b.result()
                    if result_load_b is not None and result_load_b[0] is not None and not result_load_b[0].empty:
                        loaded_dfs_for_batch.append(result_load_b)
                        logging.info(
                            f"[BATCH {batch_number}] Loaded file {i_load_b + 1}/{len(pending_files_in_batch_paths)}: {file_name_loaded_b} (Shape: {result_load_b[0].shape})")
                    else:
                        logging.warning(
                            f"[BATCH {batch_number}] Skipped during load (None DataFrame or empty): {file_name_loaded_b}")
                except Exception as e:
                    logging.error(f"[BATCH {batch_number}] Error loading file {file_name_loaded_b}: {e}",
                                  exc_info=False)
        logging.info(
            f"[BATCH {batch_number}] Finished loading {len(loaded_dfs_for_batch)} DataFrames for processing in {time.time() - loading_pool_start_time:.2f}s.")

        if not loaded_dfs_for_batch:
            logging.warning(
                f"[BATCH {batch_number}] No valid DataFrames loaded in this batch to process. Skipping to next batch.")
            continue

        proc_args_batch = [(df_proc_b, path_proc_b, max_seq_len) for df_proc_b, path_proc_b in loaded_dfs_for_batch if
                           df_proc_b is not None]

        if not proc_args_batch:
            logging.warning(
                f"[BATCH {batch_number}] No arguments for processing after filtering Nones. Skipping to next batch.")
            continue

        logging.info(
            f"[BATCH {batch_number}] Submitting {len(proc_args_batch)} loaded DataFrames for parallel processing...")
        processing_pool_start_time = time.time()
        num_process_workers = 12
        logging.info(f"[BATCH {batch_number}] Using {num_process_workers} process workers.")

        batch_results_list = []  # This will hold results from process_fragment
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
            future_to_path_map_batch = {ppool.submit(process_fragment, arg_b): arg_b[1] for arg_b in proc_args_batch}
            for k_proc_b, future_proc_b in enumerate(concurrent.futures.as_completed(future_to_path_map_batch)):
                original_file_path_b = future_to_path_map_batch[future_proc_b]
                try:
                    # process_fragment returns (file_path, pkt_arrays, label_list, mask_arrays, cat_arrays_dict_for_fragment)
                    # We need all these to save the shard correctly.
                    result_from_fragment = future_proc_b.result()
                    batch_results_list.append(result_from_fragment)  # Store the full tuple
                    logging.debug(
                        f"[BATCH {batch_number}] Processed result {k_proc_b + 1}/{len(proc_args_batch)} for: {Path(original_file_path_b).name}")
                except Exception as exc_b:
                    logging.error(
                        f"[BATCH {batch_number}] File {Path(original_file_path_b).name} generated an exception during process_fragment: {exc_b}",
                        exc_info=True)

        logging.info(
            f"[BATCH {batch_number}] Finished processing {len(batch_results_list)} DataFrames in {time.time() - processing_pool_start_time:.2f}s.")

        # Now, save the results from batch_results_list as shards
        for file_path_processed, pkt_arrs, lbls, masks, cat_arrs_dict_frag in batch_results_list:
            if pkt_arrs is None or not pkt_arrs:  # Check if process_fragment returned valid data
                logging.warning(
                    f"[BATCH {batch_number}] No sequences generated for {Path(file_path_processed).name}, skipping shard save.")
                continue

            target_shard_path = CHECKPOINT_DIR / (Path(file_path_processed).stem + ".pt")
            if target_shard_path.exists():  # Should ideally not happen if pending_files logic is correct
                logging.info(
                    f"[BATCH {batch_number}] Shard {target_shard_path.name} already exists (unexpected). Overwriting.")

            try:
                # save_shard_start_time = time.time() # Optional timing for shard save
                valid_cat_arrs_shard = {}
                num_sequences_in_shard_save = len(pkt_arrs)

                for col_name_cat_save_s in categorical_columns_packets:
                    col_data_list_s = cat_arrs_dict_frag.get(col_name_cat_save_s)
                    if col_data_list_s and all(isinstance(arr_s, np.ndarray) for arr_s in col_data_list_s) and len(
                            col_data_list_s) == num_sequences_in_shard_save:
                        stacked_cat_col_s = np.stack(col_data_list_s)
                        if stacked_cat_col_s.shape[0] == num_sequences_in_shard_save:  # Ensure first dim matches
                            valid_cat_arrs_shard[col_name_cat_save_s] = torch.from_numpy(stacked_cat_col_s).long()
                        else:  # Fallback if stacking failed or shape mismatch
                            logging.warning(
                                f"Shape mismatch for cat col {col_name_cat_save_s} in {Path(file_path_processed).name}. Expected {num_sequences_in_shard_save} sequences, got {stacked_cat_col_s.shape[0]}. Using zeros.")
                            valid_cat_arrs_shard[col_name_cat_save_s] = torch.zeros(
                                (num_sequences_in_shard_save, max_seq_len), dtype=torch.long)
                    else:  # If data is missing or malformed for this cat feature
                        logging.debug(
                            f"Missing or malformed cat data for {col_name_cat_save_s} in {Path(file_path_processed).name}. Using zeros.")
                        valid_cat_arrs_shard[col_name_cat_save_s] = torch.zeros(
                            (num_sequences_in_shard_save, max_seq_len), dtype=torch.long)

                stacked_pkt_arrs_save = np.stack(pkt_arrs) if pkt_arrs else np.array([])
                stacked_masks_save = np.stack(masks) if masks else np.array([])

                shard_to_save = {
                    "packet_seq": torch.from_numpy(stacked_pkt_arrs_save).float(),
                    "label": torch.tensor(lbls, dtype=torch.long),
                    "attention_mask": torch.from_numpy(stacked_masks_save).float(),
                    **valid_cat_arrs_shard
                }
                torch.save(shard_to_save, target_shard_path)
                processed_shards_count += 1
                logging.info(
                    f"[BATCH {batch_number}][CKPT] Wrote shard for {Path(file_path_processed).name} ({num_sequences_in_shard_save} sequences) to {target_shard_path}")
            except Exception as e:
                logging.error(
                    f"[BATCH {batch_number}] Error creating or saving shard for {Path(file_path_processed).name}: {e}",
                    exc_info=True)

        loaded_dfs_for_batch.clear()  # Clear DataFrames for this batch
        batch_results_list.clear()  # Clear results for this batch
        logging.info(f"[BATCH {batch_number}] Cleared DataFrames and results from memory for this batch.")
        logging.info(
            f"--- End of Batch {batch_number}/{total_batches}. Total shards ensured/created so far: {processed_shards_count} ---")

    # --- Memory-Efficient Aggregation Phase ---
    logging.info(
        f"\n--- Starting Memory-Efficient Aggregation from {processed_shards_count} Shard Files in {CHECKPOINT_DIR} ---")
    aggregation_start_time = time.time()

    all_shard_files = sorted(CHECKPOINT_DIR.glob("*.pt"))

    if not all_shard_files:
        logging.error(f"No shard files found in {CHECKPOINT_DIR}. Cannot aggregate. Ensure previous steps completed.")
        return

    # Initialize aggregated_data dictionary to hold the growing concatenated tensors
    aggregated_data: Dict[str, Optional[torch.Tensor]] = {
        "packet_seq": None,
        "label": None,
        "attention_mask": None,
    }
    for col in categorical_columns_packets:  # Initialize keys for all categorical columns
        aggregated_data[col] = None

    expected_keys_in_shard = ["packet_seq", "label", "attention_mask"] + categorical_columns_packets
    num_numerical_features = len(numerical_columns_packets)

    for shard_idx, shard_file_path in enumerate(all_shard_files):
        logging.info(f"Aggregating shard {shard_idx + 1}/{len(all_shard_files)}: {shard_file_path.name}")
        try:
            current_shard_data = torch.load(shard_file_path, map_location='cpu')  # Load to CPU

            for key in expected_keys_in_shard:
                if key not in current_shard_data:
                    logging.warning(
                        f"Key '{key}' not found in shard {shard_file_path.name}. Will create empty tensor if needed.")
                    # Ensure the key exists in aggregated_data if it's the first time seeing it
                    if key not in aggregated_data: aggregated_data[key] = None
                    continue  # Skip to next key for this shard

                tensor_to_append = current_shard_data[key]
                if not isinstance(tensor_to_append, torch.Tensor):
                    logging.warning(
                        f"Data for key '{key}' in shard {shard_file_path.name} is not a tensor (type: {type(tensor_to_append)}). Skipping.")
                    continue

                # If the tensor is empty (e.g. 0 sequences in this shard for this key), skip concatenation for this key
                if tensor_to_append.numel() == 0:
                    logging.debug(
                        f"Key '{key}' in shard {shard_file_path.name} is an empty tensor. Skipping concatenation for this key.")
                    if aggregated_data.get(key) is None:  # If main aggregate is also None, initialize it based on type
                        if key == "packet_seq":
                            aggregated_data[key] = torch.empty((0, max_seq_len, num_numerical_features),
                                                               dtype=torch.float32)
                        elif key == "label":
                            aggregated_data[key] = torch.empty((0,), dtype=torch.long)
                        elif key == "attention_mask":
                            aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.float32)
                        elif key in categorical_columns_packets:
                            aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.long)
                    continue

                if aggregated_data.get(key) is None or aggregated_data[
                    key].numel() == 0:  # If first tensor or previous was empty
                    aggregated_data[key] = tensor_to_append
                else:
                    try:
                        aggregated_data[key] = torch.cat((aggregated_data[key], tensor_to_append), dim=0)
                    except RuntimeError as e:
                        logging.error(f"RuntimeError concatenating key '{key}' from shard {shard_file_path.name}: {e}. "
                                      f"Aggregated shape: {aggregated_data[key].shape}, Shard tensor shape: {tensor_to_append.shape}",
                                      exc_info=True)
                        # Optionally, decide how to handle this (e.g., skip this tensor, re-initialize key)
                        # For now, we'll let it error out or try to continue if other keys are fine
                        continue

            del current_shard_data  # Crucial: free memory of the loaded shard
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache if tensors were inadvertently moved/copied

        except Exception as e:
            logging.error(f"Error processing shard {shard_file_path.name} during aggregation: {e}", exc_info=True)
            continue  # Continue to the next shard

    # Final check and default empty tensor creation for any keys that remained None
    # (e.g. if all shards were empty for a particular key)
    num_total_sequences_final = 0
    if aggregated_data.get("label") is not None and aggregated_data["label"].numel() > 0:
        num_total_sequences_final = aggregated_data["label"].shape[0]
    else:  # If label is None or empty, it means no valid sequences were aggregated
        logging.warning("No valid 'label' data aggregated. Output might be empty or inconsistent.")
        # Create empty tensors for all expected keys to ensure output file structure
        aggregated_data["packet_seq"] = torch.empty((0, max_seq_len, num_numerical_features), dtype=torch.float32)
        aggregated_data["label"] = torch.empty((0,), dtype=torch.long)
        aggregated_data["attention_mask"] = torch.empty((0, max_seq_len), dtype=torch.float32)
        for col in categorical_columns_packets:
            aggregated_data[col] = torch.empty((0, max_seq_len), dtype=torch.long)

        # Save this empty structure and return
        torch.save(aggregated_data, output_file)
        logging.info(f"[DONE] Merged dataset (empty structure) saved to {output_file} as no valid data was aggregated.")
        return

    # Ensure all expected keys have a tensor, even if it's empty, based on num_total_sequences_final
    if aggregated_data.get("packet_seq") is None or aggregated_data[
        "packet_seq"].numel() == 0 and num_total_sequences_final > 0:
        logging.warning(
            f"Packet sequence data missing or empty post-aggregation, but {num_total_sequences_final} labels exist. Creating zeros.")
        aggregated_data["packet_seq"] = torch.zeros((num_total_sequences_final, max_seq_len, num_numerical_features),
                                                    dtype=torch.float32)

    if aggregated_data.get("attention_mask") is None or aggregated_data[
        "attention_mask"].numel() == 0 and num_total_sequences_final > 0:
        logging.warning(
            f"Attention mask data missing or empty post-aggregation, but {num_total_sequences_final} labels exist. Creating ones (or zeros based on policy).")
        # Defaulting to ones, assuming valid sequences if labels exist. Adjust if padding should be indicated.
        aggregated_data["attention_mask"] = torch.ones((num_total_sequences_final, max_seq_len), dtype=torch.float32)

    for col in categorical_columns_packets:
        if aggregated_data.get(col) is None or aggregated_data[col].numel() == 0 and num_total_sequences_final > 0:
            logging.warning(
                f"Categorical data for '{col}' missing or empty post-aggregation, but {num_total_sequences_final} labels exist. Creating zeros.")
            aggregated_data[col] = torch.zeros((num_total_sequences_final, max_seq_len), dtype=torch.long)

    final_output_dict_to_save = {k: v for k, v in aggregated_data.items() if v is not None}

    logging.info(f"Aggregation completed in {time.time() - aggregation_start_time:.2f}s.")
    save_output_start_time = time.time()
    torch.save(final_output_dict_to_save, output_file)
    logging.info(f"[DONE] Merged dataset saved to {output_file} in {time.time() - save_output_start_time:.2f}s")

    final_label_tensor = final_output_dict_to_save.get("label")
    if final_label_tensor is not None and final_label_tensor.numel() > 0:
        logging.info(f"   Total sequences in this output file: {final_label_tensor.shape[0]}")
        try:
            label_counts_np_final = np.bincount(final_label_tensor.cpu().numpy())
            logging.info("Final Label distribution in this merged dataset:")
            for lbl_id_report_final, cnt_report_final in enumerate(label_counts_np_final):
                if cnt_report_final > 0:
                    label_name_report_final = next((name_report for name_report, val_report in LABEL_MAPPING.items() if
                                                    val_report == lbl_id_report_final), "Unknown")
                    logging.info(
                        f"  Class '{label_name_report_final}' ({lbl_id_report_final}): {cnt_report_final} sequences")
        except Exception as e:
            logging.error(f"Error reporting label distribution: {e}")
    else:
        logging.warning("No 'label' data in final output to report distribution or final output is empty.")

    logging.info(f"--- create_packet_sequences Finished for this run ---")


def find_label_from_path(file_path: str) -> int:
    current_path = Path(file_path).parent
    # Make sure we don't go beyond the root or a sensible base for dataset structure
    # This loop could be made safer if `dataset_dir` was passed and used as a stop condition.
    # For now, relying on `current_path != current_path.parent`
    while current_path != current_path.parent:
        folder_name = current_path.name
        for key, value in LABEL_MAPPING.items():
            if key.lower() in folder_name.lower():  # Case-insensitive match
                return value
        current_path = current_path.parent
    return -1
