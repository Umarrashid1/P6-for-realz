import os
import glob
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict  # Added Dict
import logging  # Standard logging import
import datetime
import time  # For timing

from .config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from utils import io_utils
from utils import category_mapping

# ── Set up timestamp and logging (Reverted to original behavior) ─────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"preprocessing_{timestamp}.log"  # This file will be created by this script

logging.basicConfig(  # This script will configure its own logging
    filename=LOG_FILE,
    filemode="w",
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",  # Added filename and lineno
)
# All logging calls will now use the root logger configured by basicConfig above

# ── Create a timestamped checkpoint directory ──────────────────────────────
CHECKPOINT_DIR_BASE = Path("checkpoints")
# CHECKPOINT_DIR will be set within create_packet_sequences using the global timestamp

# ── Load global numeric stats ───────────────────────────────────────────────
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
    GLOBAL_MEAN, GLOBAL_STD, GLOBAL_MEDIAN, CLIP_LOW, CLIP_HIGH = {}, {}, {}, {}, {}
    STD_COLS = []
    EPS = 1e-6


# ──────────────────────────────────────────────────────────────────────────────

def process_fragment(args: Tuple[pd.DataFrame, str, int, Path]) -> Tuple[
    str, List[np.ndarray], List[int], List[np.ndarray], Dict[str, List[np.ndarray]]]:
    df, file_path, max_seq_len, current_checkpoint_dir = args

    logging.info(f"[{Path(file_path).name}] PROCESS_FRAGMENT_START. Initial df rows: {len(df)}")
    frag_time_start = time.time()

    label = find_label_from_path(file_path)
    if label == -1:
        logging.warning(f"[{Path(file_path).name}] SKIP: No label found.")
        return file_path, [], [], [], {}

    required_packet_seq_cols = ["stream"]
    required_cols_set = set(numerical_columns_packets) | set(categorical_columns_packets) | set(
        required_packet_seq_cols)
    required_cols = list(required_cols_set)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logging.warning(f"[{Path(file_path).name}] SKIP: Missing columns: {missing_cols}")
        return file_path, [], [], [], {}

    df = df[required_cols].copy()
    logging.debug(f"[{Path(file_path).name}] Selected required columns. df shape: {df.shape}")

    num_proc_start = time.time()
    if not STD_COLS:
        logging.error(
            f"[{Path(file_path).name}] Standardization stats not loaded. Numerical preprocessing will be incomplete/incorrect.")

    for col in numerical_columns_packets:
        if col in df.columns and col in STD_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].clip(lower=CLIP_LOW[col], upper=CLIP_HIGH[col])
            df[col] = df[col].fillna(GLOBAL_MEDIAN[col])
            df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)
        elif col not in STD_COLS and col in df.columns:
            logging.warning(
                f"[{Path(file_path).name}] No standardization stats for numerical column '{col}'. Skipping its standardization, filling NaNs with 0.")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    logging.info(f"[{Path(file_path).name}] Numerical processing completed in {time.time() - num_proc_start:.2f}s.")

    cat_proc_start = time.time()
    try:
        cat_mappings = category_mapping.load_mappings(is_flow=False)
    except Exception as e:
        logging.error(
            f"[{Path(file_path).name}] CRITICAL: Failed to load category_mappings_packets.json: {e}. Assigning default 0 to categorical columns.")
        for col in categorical_columns_packets:
            if col in df.columns: df[col] = 0
        cat_mappings = {}

    for col in categorical_columns_packets:
        col_map_time_start = time.time()
        if col not in df.columns:
            logging.warning(f"[{Path(file_path).name}] Categorical column {col} unexpectedly missing. Filling with 0.")
            df[col] = 0
            continue

        mapping = cat_mappings.get(col)
        if mapping is None:
            logging.warning(f"[{Path(file_path).name}] No mapping for categorical column {col}. Filling with 0.")
            df[col] = 0
            continue

        unknown_id = mapping.get("unknown")
        if unknown_id is None:
            logging.warning(f"[{Path(file_path).name}] 'unknown' key not in mapping for {col}. Using 0 as unknown_id.")
            unknown_id = 0

        try:
            mapped_series = df[col].map(mapping)
            df[col] = mapped_series.fillna(unknown_id).astype(int)
        except TypeError as te:
            logging.error(
                f"[{Path(file_path).name}] TypeError during .map() for column {col}: {te}. Trying .apply() as fallback.")
            df[col] = df[col].apply(lambda x: mapping.get(x, unknown_id)).astype(int)  # Fallback
        except Exception as e:
            logging.error(f"[{Path(file_path).name}] Error mapping column {col}: {e}. Filling with unknown_id.")
            df[col] = unknown_id

        logging.debug(
            f"[{Path(file_path).name}] Mapped column {col} in {time.time() - col_map_time_start:.4f}s. Unique values after map: {df[col].nunique() if not df.empty else 'N/A'}")

    logging.info(f"[{Path(file_path).name}] Categorical processing completed in {time.time() - cat_proc_start:.2f}s.")

    group_time_start = time.time()
    try:
        packet_groups = df.groupby("stream", sort=False, observed=True)
    except Exception as e:
        logging.error(f"[{Path(file_path).name}] Error during groupby('stream'): {e}. Skipping sequence generation.")
        return file_path, [], [], [], {}

    num_groups = len(packet_groups) if hasattr(packet_groups, '__len__') else packet_groups.ngroups
    logging.info(
        f"[{Path(file_path).name}] Grouped by 'stream' in {time.time() - group_time_start:.2f}s. Number of groups: {num_groups}")

    seq_gen_time_start = time.time()
    pkt_arrays: List[np.ndarray] = []
    label_list: List[int] = []
    mask_arrays: List[np.ndarray] = []
    cat_arrays: Dict[str, List[np.ndarray]] = {c: [] for c in categorical_columns_packets}
    sequences_generated_count = 0

    for stream_id, packet_seq_df in packet_groups:
        if packet_seq_df.empty:
            continue

        if sequences_generated_count < 5 or sequences_generated_count % 1000 == 0:
            logging.debug(f"[{Path(file_path).name}] Processing stream_id: {stream_id}, length: {len(packet_seq_df)}")

        num_feat = packet_seq_df[numerical_columns_packets].values
        cat_feat_for_stream = {c: packet_seq_df[c].values for c in categorical_columns_packets if c in packet_seq_df}
        packet_seq_len = len(num_feat)

        def pad_and_append(start_idx, end_idx):
            nonlocal sequences_generated_count
            num_slice = num_feat[start_idx:end_idx]
            pkt_arrays.append(num_slice.astype(np.float32))

            current_slice_len = len(num_slice)
            mask = np.ones(max_seq_len, dtype=np.float32)
            if current_slice_len < max_seq_len:
                mask[current_slice_len:] = 0
            mask_arrays.append(mask)
            label_list.append(label)

            for col_name in categorical_columns_packets:
                col_mapping = cat_mappings.get(col_name, {})
                unknown_id_for_padding = col_mapping.get("unknown", 0)

                cat_original_slice = cat_feat_for_stream.get(col_name, np.array([]))[start_idx:end_idx]
                if cat_original_slice.ndim > 1: cat_original_slice = cat_original_slice.flatten()

                padded_cat_slice = np.pad(cat_original_slice, (0, max_seq_len - len(cat_original_slice)),
                                          mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays[col_name].append(padded_cat_slice.astype(np.int64))
            sequences_generated_count += 1

        if packet_seq_len == 0:
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
                col_mapping = cat_mappings.get(col_name, {})
                unknown_id_for_padding = col_mapping.get("unknown", 0)

                original_cat_data = cat_feat_for_stream.get(col_name, np.array([]))
                if original_cat_data.ndim > 1: original_cat_data = original_cat_data.flatten()

                padded_cat_col = np.pad(original_cat_data, (0, pad_len),
                                        mode='constant', constant_values=unknown_id_for_padding)
                cat_arrays[col_name].append(padded_cat_col.astype(np.int64))
            sequences_generated_count += 1
        else:
            stride = max_seq_len // 2
            if stride == 0: stride = 1

            for start in range(0, packet_seq_len - max_seq_len + 1, stride):
                pad_and_append(start, start + max_seq_len)

            if (packet_seq_len - max_seq_len) % stride != 0 and packet_seq_len > max_seq_len:
                start_of_last_segment = packet_seq_len - max_seq_len
                pad_and_append(start_of_last_segment, packet_seq_len)

    logging.info(
        f"[{Path(file_path).name}] Sequence generation loop completed in {time.time() - seq_gen_time_start:.2f}s. Generated {sequences_generated_count} sequences.")
    logging.info(
        f"[{Path(file_path).name}] PROCESS_FRAGMENT_END. Total time: {time.time() - frag_time_start:.2f}s. Output sequences: {len(pkt_arrays)}")
    return file_path, pkt_arrays, label_list, mask_arrays, cat_arrays


# ──────────────────────────────────────────────────────────────────────────────

def create_packet_sequences(
        dataset_dir: str,
        output_file: str,
        test_mode: bool = False,
        rows_per_file: int = 0,
        max_seq_len: int = 64,
):
    global timestamp
    current_checkpoint_dir = CHECKPOINT_DIR_BASE / timestamp
    current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using checkpoint directory for this run: {current_checkpoint_dir}")

    logging.info(f"--- Starting create_packet_sequences ---")
    logging.info(f"Dataset dir: {dataset_dir}, Output file: {output_file}")
    logging.info(f"Test mode: {test_mode}, Rows per file: {rows_per_file}, Max seq len: {max_seq_len}")

    effective_rows_per_file = rows_per_file
    if not test_mode and rows_per_file == 0:
        effective_rows_per_file = None
        logging.info("Processing all rows per file (test_mode=False, rows_per_file=0).")
    elif test_mode and rows_per_file == 0:
        logging.warning(
            "Test mode is True but rows_per_file is 0. This will read all rows. Consider setting rows_per_file > 0 for test mode.")
        effective_rows_per_file = None

    all_files = io_utils.list_csv_files(dataset_dir)
    logging.info(f"[START] Found {len(all_files)} CSV files.")

    pending_files = [f for f in all_files if not (current_checkpoint_dir / (Path(f).stem + ".pt")).exists()]
    if len(pending_files) < len(all_files):
        logging.info(
            f"[CHECKPOINT] {len(all_files) - len(pending_files)} files appear processed from this session (checkpoints in {current_checkpoint_dir}).")
    logging.info(f"[CHECKPOINT] {len(pending_files)} files pending processing for this run.")

    if not pending_files:
        logging.info(
            "No files pending processing. Aggregating existing shards if any from this session's checkpoint dir.")
    else:
        loading_pool_start_time = time.time()
        loaded_dfs_with_paths = []
        num_load_workers = os.cpu_count() * 2 if os.cpu_count() else 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
            futures_load = {
                tpool.submit(io_utils.load_csv_file, fp, test_mode, effective_rows_per_file): Path(fp).name
                for fp in pending_files
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures_load)):
                file_name = futures_load[future]
                try:
                    result = future.result()
                    if result is not None and result[0] is not None:
                        loaded_dfs_with_paths.append(result)
                        logging.info(
                            f"Loaded file {i + 1}/{len(pending_files)}: {file_name} (Shape: {result[0].shape})")
                    else:
                        logging.warning(f"Skipped file (None DataFrame or result from load): {file_name}")
                except Exception as e:
                    logging.error(f"Error loading file {file_name}: {e}", exc_info=False)
        logging.info(
            f"Finished loading {len(loaded_dfs_with_paths)} DataFrames in {time.time() - loading_pool_start_time:.2f}s.")

        if not loaded_dfs_with_paths and not list(current_checkpoint_dir.glob("*.pt")):
            logging.error("No CSV files could be loaded and no checkpoints found. Exiting.")
            raise RuntimeError("No files loaded and no checkpoints found for this run.")

        proc_args = [(df, path, max_seq_len, current_checkpoint_dir) for df, path in loaded_dfs_with_paths if
                     df is not None]
        if not proc_args:
            logging.warning("No valid DataFrames to process after loading stage.")
        else:
            logging.info(f"Submitting {len(proc_args)} loaded DataFrames for parallel processing...")
            processing_pool_start_time = time.time()
            num_process_workers = os.cpu_count() if os.cpu_count() else 1
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
                results_iter = ppool.map(process_fragment, proc_args, chunksize=1)

                processed_count = 0
                for file_path_processed, pkt_arrs, lbls, masks, cat_arrs in results_iter:
                    processed_count += 1
                    logging.info(
                        f"Processing result {processed_count}/{len(proc_args)} for: {Path(file_path_processed).name}")
                    if not pkt_arrs:
                        logging.warning(
                            f"No sequences generated for {Path(file_path_processed).name}, skipping shard save.")
                        continue

                    try:
                        shard_dict_start_time = time.time()
                        shard = {
                            "packet_seq": torch.from_numpy(np.stack(pkt_arrs)).float(),
                            "label": torch.tensor(lbls, dtype=torch.long),
                            "attention_mask": torch.from_numpy(np.stack(masks)).float(),
                        }
                        for col_name_cat in categorical_columns_packets:
                            if col_name_cat in cat_arrs and cat_arrs[col_name_cat]:
                                shard[col_name_cat] = torch.from_numpy(np.stack(cat_arrs[col_name_cat])).long()
                            else:
                                logging.warning(
                                    f"No data for cat column '{col_name_cat}' in shard for {Path(file_path_processed).name}. Creating empty.")
                                num_seq_in_this_shard = len(pkt_arrs)
                                shard[col_name_cat] = torch.empty((num_seq_in_this_shard, max_seq_len),
                                                                  dtype=torch.long)

                        logging.debug(
                            f"Shard dictionary for {Path(file_path_processed).name} created in {time.time() - shard_dict_start_time:.4f}s")
                        save_shard_start_time = time.time()
                        torch.save(shard, current_checkpoint_dir / (Path(file_path_processed).stem + ".pt"))
                        logging.info(
                            f"[CKPT] wrote shard for {Path(file_path_processed).name} in {time.time() - save_shard_start_time:.2f}s")
                    except Exception as e:
                        logging.error(f"Error creating or saving shard for {Path(file_path_processed).name}: {e}",
                                      exc_info=True)
            logging.info(
                f"Finished processing all submitted DataFrames in {time.time() - processing_pool_start_time:.2f}s.")

    logging.info(f"Starting aggregation of shards from: {current_checkpoint_dir}")
    aggregation_start_time = time.time()
    pkt_tensors, lbl_tensors, msk_tensors = [], [], []
    cat_tensors: Dict[str, List[torch.Tensor]] = {col: [] for col in categorical_columns_packets}
    shard_files = sorted(list(current_checkpoint_dir.glob("*.pt")))
    logging.info(f"Found {len(shard_files)} shard files for aggregation.")

    if not shard_files:
        logging.error(
            f"No shard files found in {current_checkpoint_dir} to aggregate. Cannot create output file: {output_file}")
        return

    for i, sf_path in enumerate(shard_files):
        logging.debug(f"Loading shard {i + 1}/{len(shard_files)}: {sf_path.name}")
        try:
            data = torch.load(sf_path)
            if "packet_seq" in data and data["packet_seq"].numel() > 0:
                pkt_tensors.append(data["packet_seq"])
                lbl_tensors.append(data["label"])
                msk_tensors.append(data["attention_mask"])
                for col in categorical_columns_packets:
                    if col in data and data[col].numel() > 0:
                        cat_tensors[col].append(data[col])
                    elif col in data and data[col].ndim == 2:
                        cat_tensors[col].append(data[col])
                    else:
                        logging.warning(f"Cat column '{col}' missing/malformed in shard {sf_path.name}. Placeholder.")
                        num_seq_in_shard = data["packet_seq"].shape[0] if "packet_seq" in data else 0
                        cat_tensors[col].append(torch.zeros((num_seq_in_shard, max_seq_len), dtype=torch.long))
            else:
                logging.warning(f"Skipping shard {sf_path.name} as 'packet_seq' is missing or empty.")
        except Exception as e:
            logging.error(f"Error loading/processing shard {sf_path.name} for aggregation: {e}", exc_info=True)
            continue

    if not pkt_tensors:
        logging.error(f"No valid data in any shards from {current_checkpoint_dir}. Cannot create output: {output_file}")
        return

    final_output_dict: Dict[str, torch.Tensor] = {}
    try:
        final_output_dict["packet_seq"] = torch.cat(pkt_tensors, dim=0)
        final_output_dict["label"] = torch.cat(lbl_tensors, dim=0)
        final_output_dict["attention_mask"] = torch.cat(msk_tensors, dim=0)

        for col in categorical_columns_packets:
            if cat_tensors[col]:
                final_output_dict[col] = torch.cat(cat_tensors[col], dim=0)
            else:
                logging.warning(f"No data for cat column '{col}'. Creating empty tensor in final output.")
                num_total_sequences = final_output_dict["packet_seq"].shape[
                    0] if "packet_seq" in final_output_dict else 0
                final_output_dict[col] = torch.empty((num_total_sequences, max_seq_len), dtype=torch.long)
    except Exception as e:
        logging.error(f"Error during final tensor concatenation: {e}", exc_info=True)
        return

    logging.info(f"Aggregation completed in {time.time() - aggregation_start_time:.2f}s.")
    save_output_start_time = time.time()
    torch.save(final_output_dict, output_file)
    logging.info(f"[DONE] Merged dataset saved to {output_file} in {time.time() - save_output_start_time:.2f}s")

    try:
        if "label" in final_output_dict and final_output_dict["label"].numel() > 0:
            label_counts = np.bincount(final_output_dict["label"].cpu().numpy())
            logging.info("Final Label distribution in merged dataset:")
            for lbl_id, cnt in enumerate(label_counts):
                if cnt > 0:
                    label_name = next((name for name, val in LABEL_MAPPING.items() if val == lbl_id), "Unknown")
                    logging.info(f"  Class '{label_name}' ({lbl_id}): {cnt} sequences")
        else:
            logging.warning("No 'label' data in final output to report distribution.")
    except Exception as e:
        logging.error(f"Error reporting label distribution: {e}")

    logging.info(f"--- create_packet_sequences Finished ---")


# ──────────────────────────────────────────────────────────────────────────────

def find_label_from_path(file_path: str) -> int:
    current_path = Path(file_path).parent
    while current_path != current_path.parent:
        folder_name = current_path.name
        for key, value in LABEL_MAPPING.items():
            if key.lower() in folder_name.lower():
                return value
        current_path = current_path.parent
    return -1
