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
logging.info(f"Using checkpoint directory: {CHECKPOINT_DIR.resolve()}")

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
        f"Standardization stats file not found at {STD_STATS_PATH}. Preprocessing cannot continue correctly.")
    raise SystemExit(f"Standardization stats file not found at {STD_STATS_PATH}")


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

            unknown_id = mapping.get("unknown")
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
            if num_slice.ndim == 1:
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
        output_file: str,  # This will now be a base name for part files
        test_mode: bool = False,
        rows_per_file: int = 0,
        max_seq_len: int = 64,
        files_per_processing_batch: int = 30
        # shards_per_output_file parameter removed, will be calculated for 2 parts
):
    logging.info(f"--- Starting create_packet_sequences (Packet Data) ---")
    logging.info(f"Dataset dir: {dataset_dir}, Output base for parts: {output_file}")
    logging.info(f"Test mode: {test_mode}, Rows per file (0 for all): {rows_per_file}, Max seq len: {max_seq_len}")
    logging.info(f"Files per internal processing batch: {files_per_processing_batch}")
    logging.info(f"Aggregation will produce two output part files.")

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

    processed_shards_this_run_count = 0

    # Ensure the output directory exists
    for i in range(0, len(all_files_full_list), files_per_processing_batch):
        current_batch_file_paths = all_files_full_list[i:i + files_per_processing_batch]
        batch_number = (i // files_per_processing_batch) + 1
        total_file_processing_batches = (
                                                    len(all_files_full_list) + files_per_processing_batch - 1) // files_per_processing_batch

        logging.info(
            f"\nEnsuring Shards Exist: Batch {batch_number}/{total_file_processing_batches} ({len(current_batch_file_paths)} files)")

        pending_files_in_batch_paths = []
        for fp_in_batch in current_batch_file_paths:
            shard_path = CHECKPOINT_DIR / (Path(fp_in_batch).stem + ".pt")
            if shard_path.exists():
                logging.debug(f"[BATCH {batch_number}] Checkpoint for {Path(fp_in_batch).name} already exists.")
                continue
            pending_files_in_batch_paths.append(fp_in_batch)

        # Check if there are any files that need processing in this batch
        if not pending_files_in_batch_paths:
            logging.info(f"[BATCH {batch_number}] All files in this batch already have shards.")
            continue

        logging.info(
            f"[BATCH {batch_number}] {len(pending_files_in_batch_paths)} files in this batch require shard creation.")

        loading_pool_start_time = time.time()
        loaded_dfs_for_batch = []

        num_load_workers = 12
        logging.info(f"[BATCH {batch_number}] Using {num_load_workers} thread workers for loading CSVs.")
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
                        logging.warning(f"[BATCH {batch_number}] Skipped during load: {file_name_loaded_b}")
                except Exception as e:
                    logging.error(f"[BATCH {batch_number}] Error loading file {file_name_loaded_b}: {e}",
                                  exc_info=False)
        logging.info(
            f"[BATCH {batch_number}] Loaded {len(loaded_dfs_for_batch)} DataFrames in {time.time() - loading_pool_start_time:.2f}s.")

        if not loaded_dfs_for_batch:
            logging.warning(f"[BATCH {batch_number}] No DataFrames loaded for processing in this batch.")
            continue

        proc_args_batch = [(df_proc_b, path_proc_b, max_seq_len) for df_proc_b, path_proc_b in loaded_dfs_for_batch if
                           df_proc_b is not None]
        if not proc_args_batch:
            logging.warning(f"[BATCH {batch_number}] No valid processing arguments after filtering DataFrames.")
            continue

        logging.info(f"[BATCH {batch_number}] Submitting {len(proc_args_batch)} DataFrames for parallel processing...")

        num_process_workers = 12
        logging.info(f"[BATCH {batch_number}] Using {num_process_workers} process workers.")

        batch_results_list = []
        # Process the fragments in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
            future_to_path_map_batch = {ppool.submit(process_fragment, arg_b): arg_b[1] for arg_b in proc_args_batch}
            for k_proc_b, future_proc_b in enumerate(concurrent.futures.as_completed(future_to_path_map_batch)):
                original_file_path_b = future_to_path_map_batch[future_proc_b]
                try:
                    result_from_fragment = future_proc_b.result()
                    batch_results_list.append(result_from_fragment)
                    logging.debug(
                        f"[BATCH {batch_number}] Retrieved result {k_proc_b + 1}/{len(proc_args_batch)} for: {Path(original_file_path_b).name}")
                except Exception as exc_b:
                    logging.error(
                        f"[BATCH {batch_number}] File {Path(original_file_path_b).name} generated an exception during process_fragment: {exc_b}",
                        exc_info=True)

        logging.info(f"[BATCH {batch_number}] Finished parallel processing for {len(batch_results_list)} DataFrames.")

        for file_path_processed, pkt_arrs, lbls, masks, cat_arrs_dict_frag in batch_results_list:
            if pkt_arrs is None or not pkt_arrs:
                logging.warning(
                    f"[BATCH {batch_number}] No sequences from {Path(file_path_processed).name}, skipping shard save.")
                continue

            target_shard_path = CHECKPOINT_DIR / (Path(file_path_processed).stem + ".pt")
            try:
                valid_cat_arrs_shard = {}
                num_sequences_in_shard_save = len(pkt_arrs)
                for col_name_cat_save_s in categorical_columns_packets:
                    col_data_list_s = cat_arrs_dict_frag.get(col_name_cat_save_s)
                    if col_data_list_s and all(isinstance(arr_s, np.ndarray) for arr_s in col_data_list_s) and len(
                            col_data_list_s) == num_sequences_in_shard_save:
                        stacked_cat_col_s = np.stack(col_data_list_s)
                        if stacked_cat_col_s.shape[0] == num_sequences_in_shard_save:
                            valid_cat_arrs_shard[col_name_cat_save_s] = torch.from_numpy(stacked_cat_col_s).long()
                        else:
                            valid_cat_arrs_shard[col_name_cat_save_s] = torch.zeros(
                                (num_sequences_in_shard_save, max_seq_len), dtype=torch.long)
                    else:
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
                processed_shards_this_run_count += 1
                logging.info(
                    f"[BATCH {batch_number}][CKPT] Wrote shard for {Path(file_path_processed).name} ({num_sequences_in_shard_save} sequences) to {target_shard_path}")
            except Exception as e:
                logging.error(
                    f"[BATCH {batch_number}] Error creating or saving shard for {Path(file_path_processed).name}: {e}",
                    exc_info=True)

        loaded_dfs_for_batch.clear()
        batch_results_list.clear()
        logging.info(f"[BATCH {batch_number}] Cleared DataFrames and results from memory for this batch.")
        logging.info(
            f" End of Batch {batch_number}/{total_file_processing_batches}. Total shards newly created in this run: {processed_shards_this_run_count} ")

    logging.info(f"All input files processed. Shards are available in {CHECKPOINT_DIR} for aggregation.")

    # --- Memory-Efficient Aggregation into Two Parts ---
    all_shard_files = sorted(CHECKPOINT_DIR.glob("*.pt"))
    total_shards_to_aggregate = len(all_shard_files)

    if not all_shard_files:
        logging.error(f"No shard files found in {CHECKPOINT_DIR}. Cannot aggregate.")
        return

    logging.info(
        f"\nStarting Memory-Efficient Aggregation from {total_shards_to_aggregate} Shard Files into Two Parts.")
    aggregation_start_time = time.time()


    output_file_path_obj = Path(output_file)
    output_base_name = output_file_path_obj.stem
    output_suffix = output_file_path_obj.suffix
    output_dir = output_file_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate number of shards for the first part (roughly half, favoring first part if odd)
    shards_for_part_1 = (total_shards_to_aggregate + 1) // 2
    shards_for_part_2 = total_shards_to_aggregate - shards_for_part_1

    part_files_to_process = [
        all_shard_files[:shards_for_part_1],
        all_shard_files[shards_for_part_1:]
    ]

    expected_keys_in_shard = ["packet_seq", "label", "attention_mask"] + categorical_columns_packets
    num_numerical_features = len(numerical_columns_packets)

    # Ensure max_seq_len is defined
    for part_idx, current_part_shard_list in enumerate(part_files_to_process):
        part_num = part_idx + 1
        if not current_part_shard_list:
            logging.info(f"No shards to process for part {part_num}. Skipping.")
            continue

        logging.info(f"\nAggregating Part {part_num} ({len(current_part_shard_list)} shards)")

        current_part_aggregated_data: Dict[str, Optional[torch.Tensor]] = {
            "packet_seq": None, "label": None, "attention_mask": None,
            **{col: None for col in categorical_columns_packets}
        }
        # Initialize empty tensors for each expected key
        for shard_in_part_idx, shard_file_path in enumerate(current_part_shard_list):
            logging.info(
                f"Part {part_num}: Processing shard {shard_in_part_idx + 1}/{len(current_part_shard_list)}: {shard_file_path.name}")
            try:
                current_shard_data = torch.load(shard_file_path, map_location='cpu')
                # Check if all expected keys are present in the shard
                for key in expected_keys_in_shard:
                    if key not in current_shard_data:
                        logging.warning(f"Key '{key}' not in shard {shard_file_path.name} for part {part_num}.")
                        if current_part_aggregated_data.get(key) is None:  # Initialize if not already
                            if key == "packet_seq":
                                current_part_aggregated_data[key] = torch.empty(
                                    (0, max_seq_len, num_numerical_features), dtype=torch.float32)
                            elif key == "label":
                                current_part_aggregated_data[key] = torch.empty((0,), dtype=torch.long)
                            elif key == "attention_mask":
                                current_part_aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.float32)
                            elif key in categorical_columns_packets:
                                current_part_aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.long)
                        continue

                    tensor_to_append = current_shard_data[key]
                    if not isinstance(tensor_to_append, torch.Tensor):
                        logging.warning(f"Data for key '{key}' in shard {shard_file_path.name} not a tensor. Skipping.")
                        continue
                    if tensor_to_append.numel() == 0:
                        logging.debug(f"Key '{key}' in shard {shard_file_path.name} is empty. Skipping concat.")
                        if current_part_aggregated_data.get(key) is None:  # Initialize if not already for this part
                            if key == "packet_seq":
                                current_part_aggregated_data[key] = torch.empty(
                                    (0, max_seq_len, num_numerical_features), dtype=torch.float32)
                            elif key == "label":
                                current_part_aggregated_data[key] = torch.empty((0,), dtype=torch.long)
                            elif key == "attention_mask":
                                current_part_aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.float32)
                            elif key in categorical_columns_packets:
                                current_part_aggregated_data[key] = torch.empty((0, max_seq_len), dtype=torch.long)
                        continue

                    current_agg_tensor = current_part_aggregated_data.get(key)
                    if current_agg_tensor is None or current_agg_tensor.numel() == 0:
                        current_part_aggregated_data[key] = tensor_to_append
                    else:
                        try:
                            current_part_aggregated_data[key] = torch.cat((current_agg_tensor, tensor_to_append), dim=0)
                        except RuntimeError as e:
                            logging.error(
                                f"RuntimeError concatenating key '{key}' (Part {part_num}, Shard {shard_file_path.name}): {e}. Aggregated shape: {current_agg_tensor.shape}, Shard tensor shape: {tensor_to_append.shape}",
                                exc_info=True)
                            continue

                del current_shard_data
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error processing shard {shard_file_path.name} for part {part_num}: {e}", exc_info=True)
                continue

        # Save the current aggregated part
        current_part_output_filename = output_dir / f"{output_base_name}_part_{part_num}{output_suffix}"
        num_sequences_in_part = 0
        if current_part_aggregated_data.get("label") is not None and current_part_aggregated_data["label"].numel() > 0:
            num_sequences_in_part = current_part_aggregated_data["label"].shape[0]

        if num_sequences_in_part == 0 and not any(
                v is not None and v.numel() > 0 for v in current_part_aggregated_data.values()):
            logging.info(f"Part {part_num} is empty (no sequences aggregated). Skipping save for this part.")
        else:
            # Ensure all keys are present in the dict to be saved for this part, even if empty
            for key_check in expected_keys_in_shard:
                if current_part_aggregated_data.get(key_check) is None:
                    if key_check == "packet_seq":
                        current_part_aggregated_data[key_check] = torch.empty((0, max_seq_len, num_numerical_features),
                                                                              dtype=torch.float32)
                    elif key_check == "label":
                        current_part_aggregated_data[key_check] = torch.empty((0,), dtype=torch.long)
                    elif key_check == "attention_mask":
                        current_part_aggregated_data[key_check] = torch.empty((0, max_seq_len), dtype=torch.float32)
                    elif key_check in categorical_columns_packets:
                        current_part_aggregated_data[key_check] = torch.empty((0, max_seq_len), dtype=torch.long)

            logging.info(
                f"Saving aggregated part {part_num} to {current_part_output_filename} ({num_sequences_in_part} sequences).")
            torch.save(current_part_aggregated_data, current_part_output_filename)

        del current_part_aggregated_data  # Free memory after saving
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logging.info(
        f"--- All parts aggregated and saved. Total time for aggregation: {time.time() - aggregation_start_time:.2f}s ---")
    logging.info(f"Output part files are located in: {output_dir}")
    logging.info(f"--- create_packet_sequences Finished for this run ---")


def find_label_from_path(file_path: str) -> int:
    current_path = Path(file_path).parent
    while current_path != current_path.parent:
        folder_name = current_path.name
        for key, value in LABEL_MAPPING.items():
            if key.lower() in folder_name.lower():
                return value
        current_path = current_path.parent
    return -1