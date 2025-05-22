import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional  # Optional is still used for return types
import logging
import datetime
import time
import random  # For shuffling if needed, but not currently used for selection

from .config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from utils import io_utils  # Assuming io_utils is in utils directory relative to this
from utils import category_mapping  # Assuming category_mapping is in utils

# ── Set up logging ─────────────────
LOG_FILE_BASENAME = "preprocessing_packets"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for log file name only
LOG_FILE = f"{LOG_FILE_BASENAME}_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

# ── FIXED Checkpoint directory ─────────────────────────
CHECKPOINT_DIR = Path("checkpoints_packet_shards")  # Fixed name for resumability
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
    for col_num_proc in numerical_columns_packets:  # Renamed col to col_num_proc
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
        for col_cat_proc in categorical_columns_packets:
            if col_cat_proc in df.columns: df[col_cat_proc] = 0
        cat_mappings = {}

    for col_cat_proc in categorical_columns_packets:  # Renamed col to col_cat_proc
        col_map_time_start = time.time()
        if col_cat_proc not in df.columns:
            logging.warning(
                f"[{Path(file_path).name}] Categorical column {col_cat_proc} unexpectedly missing. Filling with 0.")
            df[col_cat_proc] = 0
            continue

        df[col_cat_proc] = df[col_cat_proc].astype(str).fillna("unknown")
        mapping = cat_mappings.get(col_cat_proc)
        unknown_id = 0  # Default unknown_id
        if mapping:
            unknown_id = mapping.get("unknown", 0)  # Get specific unknown_id if mapping and key exist
        else:
            logging.warning(
                f"[{Path(file_path).name}] No mapping found for categorical column {col_cat_proc}. Using 0 as unknown_id for all values.")

        # Robust mapping lookup
        temp_mapped_values = []
        if mapping:
            for val_in_df in df[col_cat_proc]:
                mapped_val = mapping.get(val_in_df)  # Try direct string match
                if mapped_val is None:  # Try original type if it was numeric-like
                    try:
                        mapped_val = mapping.get(int(val_in_df))
                    except ValueError:
                        pass
                if mapped_val is None:
                    try:
                        mapped_val = mapping.get(float(val_in_df))
                    except ValueError:
                        pass
                temp_mapped_values.append(mapped_val if mapped_val is not None else unknown_id)
            df[col_cat_proc] = pd.Series(temp_mapped_values, index=df.index).astype(int)
        else:  # No mapping found for this column at all
            df[col_cat_proc] = unknown_id  # Assign default unknown_id to all

        logging.debug(
            f"[{Path(file_path).name}] Mapped column {col_cat_proc} in {time.time() - col_map_time_start:.4f}s.")

    logging.info(f"[{Path(file_path).name}] Categorical processing completed in {time.time() - cat_proc_start:.2f}s.")

    if 'stream' not in df.columns:
        logging.error(f"[{Path(file_path).name}] 'stream' column missing for groupby. Cannot proceed.")
        return file_path, None, None, None, None
    packet_groups = df.groupby("stream", sort=False, observed=True)
    num_groups = len(packet_groups) if hasattr(packet_groups, '__len__') else packet_groups.ngroups
    logging.info(
        f"[{Path(file_path).name}] Grouped by 'stream'. Number of groups: {num_groups}")

    seq_gen_time_start = time.time()
    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    cat_arrays_dict_for_fragment: Dict[str, List[np.ndarray]] = {c: [] for c in
                                                                 categorical_columns_packets}  # Renamed cat_arrays to avoid confusion
    sequences_generated_count = 0

    for stream_id, packet_seq_df in packet_groups:
        if packet_seq_df.empty: continue

        num_feat_df = packet_seq_df[numerical_columns_packets]
        for col_num_seq in numerical_columns_packets:  # Renamed col to col_num_seq
            if col_num_seq not in num_feat_df.columns: num_feat_df[col_num_seq] = 0
        num_feat = num_feat_df[numerical_columns_packets].values

        cat_feat_for_stream = {}
        for c_col_seq in categorical_columns_packets:  # Renamed c to c_col_seq
            if c_col_seq in packet_seq_df:
                cat_feat_for_stream[c_col_seq] = packet_seq_df[c_col_seq].values
            else:
                cat_feat_for_stream[c_col_seq] = np.zeros(len(num_feat), dtype=np.int64)

        packet_seq_len = len(num_feat)
        if packet_seq_len == 0: continue

        def pad_and_append_segment(start_idx, end_idx):
            nonlocal sequences_generated_count
            num_slice = num_feat[start_idx:end_idx]
            padded_num_slice = np.pad(num_slice, ((0, max_seq_len - len(num_slice)), (0, 0)), mode='constant',
                                      constant_values=0)
            pkt_arrays.append(padded_num_slice.astype(np.float32))

            current_slice_len = len(num_slice)
            mask = np.ones(max_seq_len, dtype=np.float32)
            if current_slice_len < max_seq_len: mask[current_slice_len:] = 0
            mask_arrays.append(mask)
            label_list.append(label)

            for col_name_cat_seq in categorical_columns_packets:  # Renamed col_name to col_name_cat_seq
                col_mapping_seq = cat_mappings.get(col_name_cat_seq, {})  # Renamed col_mapping to col_mapping_seq
                unknown_id_for_padding = col_mapping_seq.get("unknown", 0)
                cat_original_slice = cat_feat_for_stream.get(col_name_cat_seq, np.array([]))[start_idx:end_idx]
                if cat_original_slice.ndim > 1: cat_original_slice = cat_original_slice.flatten()
                padded_cat_slice = np.pad(cat_original_slice, (0, max_seq_len - len(cat_original_slice)),
                                          mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays_dict_for_fragment[col_name_cat_seq].append(padded_cat_slice.astype(np.int64))
            sequences_generated_count += 1

        if packet_seq_len <= max_seq_len:
            pad_and_append_segment(0, packet_seq_len)
        else:
            stride = max_seq_len // 2
            if stride == 0: stride = 1
            for start in range(0, packet_seq_len - max_seq_len + 1, stride):
                pad_and_append_segment(start, start + max_seq_len)
            last_segment_start = (packet_seq_len - max_seq_len)
            # Check if the last segment (aligned to the end) is different from the last one processed by striding
            if not (
            ((packet_seq_len - max_seq_len) % stride == 0) if stride > 0 else (packet_seq_len - max_seq_len == 0)):
                if last_segment_start > (
                ((packet_seq_len - max_seq_len) // stride) * stride if stride > 0 else 0):  # Ensure it's a new segment
                    pad_and_append_segment(last_segment_start, packet_seq_len)

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
        files_per_processing_batch: int = 30  # Default internal batch size
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

    # These will accumulate data from all processed shards (new or loaded)
    final_pkt_tensors, final_lbl_tensors, final_msk_tensors = [], [], []
    final_cat_tensors_dict: Dict[str, List[torch.Tensor]] = {col: [] for col in categorical_columns_packets}

    processed_file_stems_in_run = set()  # Keep track of files processed/loaded in this run to avoid double aggregation

    for i in range(0, len(all_files_full_list), files_per_processing_batch):
        current_batch_file_paths = all_files_full_list[i:i + files_per_processing_batch]
        batch_number = (i // files_per_processing_batch) + 1
        total_batches = (len(all_files_full_list) + files_per_processing_batch - 1) // files_per_processing_batch

        logging.info(
            f"\n--- Processing Batch {batch_number}/{total_batches} ({len(current_batch_file_paths)} files) ---")

        pending_files_in_batch_paths = []
        for fp_in_batch in current_batch_file_paths:
            shard_path = CHECKPOINT_DIR / (Path(fp_in_batch).stem + ".pt")
            if shard_path.exists() and Path(fp_in_batch).stem not in processed_file_stems_in_run:
                logging.info(
                    f"[BATCH {batch_number}] Found existing checkpoint for {Path(fp_in_batch).name}. Loading it.")
                try:
                    data = torch.load(shard_path)
                    if "packet_seq" in data and data["packet_seq"].numel() > 0:
                        final_pkt_tensors.append(data["packet_seq"])
                        final_lbl_tensors.append(data["label"])
                        final_msk_tensors.append(data["attention_mask"])
                        for col_cat_load in categorical_columns_packets:  # Renamed col to col_cat_load
                            if col_cat_load in data and data[col_cat_load].numel() > 0:
                                final_cat_tensors_dict[col_cat_load].append(data[col_cat_load])
                            elif col_cat_load in data and data[col_cat_load].ndim == 2 and data[col_cat_load].shape[
                                0] == data["packet_seq"].shape[0]:
                                final_cat_tensors_dict[col_cat_load].append(data[col_cat_load])
                            else:
                                num_seq_in_shard_load = data["packet_seq"].shape[
                                    0]  # Renamed num_seq_in_shard to num_seq_in_shard_load
                                final_cat_tensors_dict[col_cat_load].append(
                                    torch.zeros((num_seq_in_shard_load, max_seq_len), dtype=torch.long))
                        processed_file_stems_in_run.add(Path(fp_in_batch).stem)
                    else:
                        logging.warning(
                            f"[BATCH {batch_number}] Existing shard {shard_path.name} is empty or lacks 'packet_seq'. Will attempt to re-process.")
                        pending_files_in_batch_paths.append(fp_in_batch)
                except Exception as e:
                    logging.error(
                        f"[BATCH {batch_number}] Error loading existing shard {shard_path.name}: {e}. Will attempt to re-process.",
                        exc_info=True)
                    pending_files_in_batch_paths.append(fp_in_batch)
            elif Path(fp_in_batch).stem not in processed_file_stems_in_run:
                pending_files_in_batch_paths.append(fp_in_batch)

        if not pending_files_in_batch_paths:
            logging.info(
                f"[BATCH {batch_number}] All files in this batch were already processed and loaded from checkpoints.")
            continue

        logging.info(
            f"[BATCH {batch_number}] {len(pending_files_in_batch_paths)} files in this batch require processing.")

        loading_pool_start_time = time.time()
        loaded_dfs_for_batch = []
        num_load_workers = min(os.cpu_count() * 2 if os.cpu_count() else 4, 32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
            futures_load_batch = {  # Renamed futures_load to futures_load_batch
                tpool.submit(io_utils.load_csv_file, fp_load_batch, test_mode, effective_rows_per_file): Path(
                    fp_load_batch).name  # Renamed fp to fp_load_batch
                for fp_load_batch in pending_files_in_batch_paths
            }
            for i_load_batch, future_load_batch in enumerate(
                    concurrent.futures.as_completed(futures_load_batch)):  # Renamed i_load, future_load
                file_name_loaded_batch = futures_load_batch[future_load_batch]  # Renamed file_name_loaded
                try:
                    result_load_batch = future_load_batch.result()  # Renamed result_load
                    if result_load_batch is not None and result_load_batch[0] is not None and not result_load_batch[
                        0].empty:
                        loaded_dfs_for_batch.append(result_load_batch)
                        logging.info(
                            f"[BATCH {batch_number}] Loaded file {i_load_batch + 1}/{len(pending_files_in_batch_paths)}: {file_name_loaded_batch} (Shape: {result_load_batch[0].shape})")
                    else:
                        logging.warning(
                            f"[BATCH {batch_number}] Skipped during load (None DataFrame or empty): {file_name_loaded_batch}")
                except Exception as e:
                    logging.error(f"[BATCH {batch_number}] Error loading file {file_name_loaded_batch}: {e}",
                                  exc_info=False)
        logging.info(
            f"[BATCH {batch_number}] Finished loading {len(loaded_dfs_for_batch)} DataFrames for processing in {time.time() - loading_pool_start_time:.2f}s.")

        if not loaded_dfs_for_batch:
            logging.warning(
                f"[BATCH {batch_number}] No valid DataFrames loaded in this batch to process. Skipping to next batch.")
            continue

        proc_args_batch = [(df_proc_batch, path_proc_batch, max_seq_len) for df_proc_batch, path_proc_batch in
                           loaded_dfs_for_batch if df_proc_batch is not None]  # Renamed df,path

        if not proc_args_batch:
            logging.warning(
                f"[BATCH {batch_number}] No arguments for processing after filtering Nones. Skipping to next batch.")
            continue

        logging.info(
            f"[BATCH {batch_number}] Submitting {len(proc_args_batch)} loaded DataFrames for parallel processing...")
        processing_pool_start_time = time.time()
        num_process_workers = os.cpu_count() if os.cpu_count() else 1

        batch_results_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
            future_to_path_map_batch = {ppool.submit(process_fragment, arg_batch): arg_batch[1] for arg_batch in
                                        proc_args_batch}  # Renamed future_to_path_map, arg
            for k_proc_batch, future_proc_batch in enumerate(
                    concurrent.futures.as_completed(future_to_path_map_batch)):  # Renamed k_proc, future_proc
                original_file_path_batch = future_to_path_map_batch[future_proc_batch]  # Renamed original_file_path
                try:
                    batch_results_list.append(future_proc_batch.result())
                    logging.debug(
                        f"[BATCH {batch_number}] Processed result {k_proc_batch + 1}/{len(proc_args_batch)} for: {Path(original_file_path_batch).name}")
                except Exception as exc_batch:  # Renamed exc
                    logging.error(
                        f"[BATCH {batch_number}] File {Path(original_file_path_batch).name} generated an exception during process_fragment: {exc_batch}",
                        exc_info=True)

        logging.info(
            f"[BATCH {batch_number}] Finished processing {len(batch_results_list)} DataFrames in {time.time() - processing_pool_start_time:.2f}s.")

        for file_path_processed, pkt_arrs, lbls, masks, cat_arrs_dict_frag in batch_results_list:  # Renamed cat_arrs_dict
            if pkt_arrs is None or not pkt_arrs:
                logging.warning(
                    f"[BATCH {batch_number}] No sequences generated for {Path(file_path_processed).name}, skipping shard save.")
                continue

            if Path(file_path_processed).stem in processed_file_stems_in_run:  # Should not happen if pending_files_in_batch was correct
                logging.warning(
                    f"[BATCH {batch_number}] Shard for {Path(file_path_processed).name} was already loaded or processed in this run. Skipping save and accumulation to avoid duplication.")
                continue

            try:
                save_shard_start_time = time.time()
                valid_cat_arrs_shard = {}  # Renamed valid_cat_arrs
                num_sequences_in_shard_save = len(pkt_arrs)  # Renamed num_sequences_in_shard

                for col_name_cat_save_shard in categorical_columns_packets:  # Renamed col_name_cat_save
                    col_data_list_shard = cat_arrs_dict_frag.get(col_name_cat_save_shard)  # Renamed col_data_list
                    if col_data_list_shard and all(
                            isinstance(arr_shard, np.ndarray) for arr_shard in col_data_list_shard) and len(
                            col_data_list_shard) == num_sequences_in_shard_save:  # Renamed arr
                        stacked_cat_col_shard = np.stack(col_data_list_shard)  # Renamed stacked_cat_col
                        if stacked_cat_col_shard.shape[0] == num_sequences_in_shard_save:
                            valid_cat_arrs_shard[col_name_cat_save_shard] = torch.from_numpy(
                                stacked_cat_col_shard).long()
                        else:
                            valid_cat_arrs_shard[col_name_cat_save_shard] = torch.zeros(
                                (num_sequences_in_shard_save, max_seq_len), dtype=torch.long)
                    else:
                        valid_cat_arrs_shard[col_name_cat_save_shard] = torch.zeros(
                            (num_sequences_in_shard_save, max_seq_len), dtype=torch.long)

                stacked_pkt_arrs_save = np.stack(pkt_arrs) if pkt_arrs else np.array([])  # Renamed stacked_pkt_arrs
                stacked_masks_save = np.stack(masks) if masks else np.array([])  # Renamed stacked_masks

                shard = {
                    "packet_seq": torch.from_numpy(stacked_pkt_arrs_save).float(),
                    "label": torch.tensor(lbls, dtype=torch.long),
                    "attention_mask": torch.from_numpy(stacked_masks_save).float(),
                    **valid_cat_arrs_shard
                }

                target_shard_path = CHECKPOINT_DIR / (Path(file_path_processed).stem + ".pt")
                torch.save(shard, target_shard_path)
                logging.info(
                    f"[BATCH {batch_number}][CKPT] Wrote shard for {Path(file_path_processed).name} ({num_sequences_in_shard_save} sequences) to {target_shard_path}")

                final_pkt_tensors.append(shard["packet_seq"])
                final_lbl_tensors.append(shard["label"])
                final_msk_tensors.append(shard["attention_mask"])
                for col_final_cat in categorical_columns_packets:  # Renamed col
                    final_cat_tensors_dict[col_final_cat].append(shard[col_final_cat])
                processed_file_stems_in_run.add(Path(file_path_processed).stem)

            except Exception as e:
                logging.error(
                    f"[BATCH {batch_number}] Error creating or saving shard for {Path(file_path_processed).name}: {e}",
                    exc_info=True)

        loaded_dfs_for_batch.clear()
        logging.info(f"[BATCH {batch_number}] Cleared DataFrames from memory for this batch.")
        logging.info(
            f"--- End of Batch {batch_number}/{total_batches}. Files processed in this run via new shards: {len(processed_file_stems_in_run)} ---")

    # --- Aggregation Phase ---
    if not final_pkt_tensors:  # If no tensors were accumulated (e.g. all loaded from existing checkpoints in first batch and no other batches, or all processing failed)
        logging.error(
            f"No data tensors accumulated or loaded to aggregate. Check if shards exist in {CHECKPOINT_DIR} or if processing steps completed. Cannot create output: {output_file}")
        return

    logging.info(
        f"Starting final aggregation of {len(final_pkt_tensors)} accumulated/loaded shard groups (representing {len(processed_file_stems_in_run)} unique files).")
    aggregation_start_time = time.time()

    final_output_dict: Dict[str, torch.Tensor] = {}
    try:
        final_output_dict["packet_seq"] = torch.cat(final_pkt_tensors, dim=0)
        final_output_dict["label"] = torch.cat(final_lbl_tensors, dim=0)
        final_output_dict["attention_mask"] = torch.cat(final_msk_tensors, dim=0)

        for col_agg_final in categorical_columns_packets:  # Renamed col_final_agg
            if final_cat_tensors_dict[col_agg_final]:
                try:
                    final_output_dict[col_agg_final] = torch.cat(final_cat_tensors_dict[col_agg_final], dim=0)
                except RuntimeError as e:
                    logging.error(f"RuntimeError concatenating categorical column '{col_agg_final}': {e}.",
                                  exc_info=True)
                    num_total_sequences_agg_err = final_output_dict["packet_seq"].shape[
                        0]  # Renamed num_total_sequences
                    final_output_dict[col_agg_final] = torch.zeros((num_total_sequences_agg_err, max_seq_len),
                                                                   dtype=torch.long)
            else:
                logging.warning(
                    f"No data accumulated for cat column '{col_agg_final}'. Creating empty tensor in final output.")
                num_total_sequences_agg_empty = final_output_dict["packet_seq"].shape[
                    0] if "packet_seq" in final_output_dict else 0  # Renamed num_total_sequences
                final_output_dict[col_agg_final] = torch.zeros((num_total_sequences_agg_empty, max_seq_len),
                                                               dtype=torch.long)

    except Exception as e:
        logging.error(f"Error during final tensor concatenation: {e}", exc_info=True)
        return

    logging.info(f"Aggregation completed in {time.time() - aggregation_start_time:.2f}s.")
    save_output_start_time = time.time()
    torch.save(final_output_dict, output_file)
    logging.info(f"[DONE] Merged dataset saved to {output_file} in {time.time() - save_output_start_time:.2f}s")
    logging.info(
        f"   Total sequences in this output file: {final_output_dict['label'].shape[0] if 'label' in final_output_dict and final_output_dict['label'].numel() > 0 else 'N/A'}")

    try:
        if "label" in final_output_dict and final_output_dict["label"].numel() > 0:
            label_counts_np_final = np.bincount(final_output_dict["label"].cpu().numpy())  # Renamed label_counts_np
            logging.info("Final Label distribution in this merged dataset:")
            for lbl_id_report_final, cnt_report_final in enumerate(
                    label_counts_np_final):  # Renamed lbl_id_report, cnt_report
                if cnt_report_final > 0:
                    label_name_report_final = next((name_report for name_report, val_report in LABEL_MAPPING.items() if
                                                    val_report == lbl_id_report_final),
                                                   "Unknown")  # Renamed name, val, label_name
                    logging.info(
                        f"  Class '{label_name_report_final}' ({lbl_id_report_final}): {cnt_report_final} sequences")
        else:
            logging.warning("No 'label' data in final output to report distribution.")
    except Exception as e:
        logging.error(f"Error reporting label distribution: {e}")

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