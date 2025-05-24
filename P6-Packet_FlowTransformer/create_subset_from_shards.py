# create_subset_from_shards.py
import os
import glob
import numpy as np
import torch
from pathlib import Path
import logging
import datetime
import time
import json  # For loading config.json
from typing import Dict, Optional  # Added for type hinting


# --- Early: Try to set up basic print-based error catching ---
def early_error_handler(exc_type, exc_value, exc_traceback):
    print(f"EARLY PYTHON EXCEPTION: Type: {exc_type}, Value: {exc_value}", flush=True)
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)


import sys

sys.excepthook = early_error_handler  # Catches unhandled exceptions early

# --- Global Configuration (Adjust paths if necessary) ---
try:
    from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Ensure pipeline.config is accessible. Exiting.", flush=True)
    sys.exit(1)

LOG_FILE_BASENAME = "subset_aggregation"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH_OBJ = Path(f"{LOG_FILE_BASENAME}_{timestamp}.log")

SHARD_CHECKPOINT_DIR = Path("checkpoints_packet_shards")
# MODIFIED: Changed filename to reflect 1/4 subset
SUBSET_OUTPUT_FILE = Path("..") / ".." / ".." / "dataset" / "packet_subset_quartered.pt"


# --- Logging Setup Function ---
def setup_logging():
    print(f"DEBUG: Entered setup_logging(). Attempting to log to: {LOG_FILE_PATH_OBJ.resolve()}", flush=True)
    try:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=str(LOG_FILE_PATH_OBJ),
            filemode="w",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            force=True
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        logging.info(f"Logging initialized. Log file: {LOG_FILE_PATH_OBJ.resolve()}")
        print(f"DEBUG: Logging setup complete. Log file should be at {LOG_FILE_PATH_OBJ.resolve()}", flush=True)
    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR during setup_logging: {e}", flush=True)
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
        logging.error(f"Fallback logging: Error during file logging setup: {e}")


# --- Main Function ---
# MODIFIED: Function name updated
def create_quartered_subset():
    logging.info(f"--- Entered create_quartered_subset function ---")

    max_s_len_for_shards = 64
    num_numerical_features = len(numerical_columns_packets)

    try:
        logging.debug("Attempting to load config.json...")
        with open("config.json", 'r') as f:
            config_data = json.load(f)
            max_s_len_for_shards = config_data['model_architecture']['max_seq_len_packet']
            num_numerical_features_from_config = config_data['model_architecture'].get('input_dim_packet_numerical',
                                                                                       num_numerical_features)
            if num_numerical_features_from_config != num_numerical_features:
                logging.warning(
                    f"Mismatch in num_numerical_features from config.json ({num_numerical_features_from_config}) and pipeline.config ({num_numerical_features}). Using value from pipeline.config.")
            logging.info(f"Successfully loaded MAX_SEQ_LEN = {max_s_len_for_shards} from config.json")
    except FileNotFoundError:
        logging.warning(
            f"config.json not found in CWD ({os.getcwd()}). Using default MAX_SEQ_LEN = {max_s_len_for_shards}.")
    except Exception as e:
        logging.warning(
            f"Error loading MAX_SEQ_LEN from config.json ({e}). Using default MAX_SEQ_LEN = {max_s_len_for_shards}.")

    logging.info(
        f"Starting Subset Creation (1/4). Shard dir: {SHARD_CHECKPOINT_DIR.resolve()}, Output: {SUBSET_OUTPUT_FILE.resolve()}, MAX_SEQ_LEN: {max_s_len_for_shards}")

    all_shard_files = sorted(SHARD_CHECKPOINT_DIR.glob("*.pt"))
    total_shards_to_process = len(all_shard_files)

    if not all_shard_files:
        logging.error(f"No shard files found in {SHARD_CHECKPOINT_DIR}. Cannot create subset.")
        return

    logging.info(f"Found {total_shards_to_process} shard files to process.")
    aggregation_start_time = time.time()

    # MODIFIED: Variable name updated
    aggregated_quartered_data: Dict[str, Optional[torch.Tensor]] = {
        "packet_seq": None, "label": None, "attention_mask": None,
        **{col: None for col in categorical_columns_packets}
    }
    expected_keys_in_shard = ["packet_seq", "label", "attention_mask"] + categorical_columns_packets

    for shard_idx, shard_file_path in enumerate(all_shard_files):
        logging.info(f"Processing shard {shard_idx + 1}/{total_shards_to_process}: {shard_file_path.name}")
        try:
            current_shard_data = torch.load(shard_file_path, map_location='cpu')
            if "label" not in current_shard_data or not isinstance(current_shard_data["label"], torch.Tensor) or \
                    current_shard_data["label"].numel() == 0:
                logging.warning(f"Shard {shard_file_path.name} missing 'label' or label is empty. Skipping.")
                continue

            num_rows_in_shard = current_shard_data["label"].shape[0]
            if num_rows_in_shard == 0:
                logging.info(f"Shard {shard_file_path.name} has 0 sequences. Skipping.")
                continue

            # MODIFIED: Changed from // 2 to // 4 for 1/4 subset
            num_rows_to_keep = num_rows_in_shard // 4
            if num_rows_to_keep == 0 and num_rows_in_shard > 0: num_rows_to_keep = 1  # Keep at least one if shard not empty
            logging.debug(
                f"  Shard {shard_file_path.name}: total {num_rows_in_shard}, keeping {num_rows_to_keep} (approx 1/4).")

            for key in expected_keys_in_shard:
                if key not in current_shard_data:
                    logging.warning(f"Key '{key}' missing in shard {shard_file_path.name}.")
                    # MODIFIED: Variable name updated
                    if aggregated_quartered_data.get(key) is None:
                        if key == "packet_seq":
                            aggregated_quartered_data[key] = torch.empty(
                                (0, max_s_len_for_shards, num_numerical_features),
                                dtype=torch.float32)
                        elif key == "label":
                            aggregated_quartered_data[key] = torch.empty((0,), dtype=torch.long)
                        elif key == "attention_mask":
                            aggregated_quartered_data[key] = torch.empty((0, max_s_len_for_shards), dtype=torch.float32)
                        elif key in categorical_columns_packets:
                            aggregated_quartered_data[key] = torch.empty((0, max_s_len_for_shards), dtype=torch.long)
                    continue

                tensor_full = current_shard_data[key]
                if not isinstance(tensor_full, torch.Tensor) or tensor_full.shape[0] == 0:
                    logging.warning(
                        f"Key '{key}' in shard {shard_file_path.name} is not a tensor or has 0 sequences. Shape: {tensor_full.shape if isinstance(tensor_full, torch.Tensor) else 'Not a Tensor'}. Skipping.")
                    # MODIFIED: Variable name updated
                    if aggregated_quartered_data.get(key) is None:
                        if key == "packet_seq":
                            aggregated_quartered_data[key] = torch.empty(
                                (0, max_s_len_for_shards, num_numerical_features),
                                dtype=torch.float32)
                        elif key == "label":
                            aggregated_quartered_data[key] = torch.empty((0,), dtype=torch.long)
                        elif key == "attention_mask":
                            aggregated_quartered_data[key] = torch.empty((0, max_s_len_for_shards), dtype=torch.float32)
                        elif key in categorical_columns_packets:
                            aggregated_quartered_data[key] = torch.empty((0, max_s_len_for_shards), dtype=torch.long)
                    continue

                current_key_rows = tensor_full.shape[0]
                rows_for_this_key = min(num_rows_to_keep, current_key_rows)

                # MODIFIED: Logic to handle rows_for_this_key if it was 0 due to num_rows_to_keep
                # but the current key actually has data. This ensures we attempt to take 1/4 of this key's data.
                if rows_for_this_key == 0 and current_key_rows > 0:
                    rows_for_this_key = current_key_rows // 4  # Take 1/4 of this specific key
                    if rows_for_this_key == 0: rows_for_this_key = 1  # Ensure at least 1 if possible

                if rows_for_this_key == 0:
                    logging.debug(f"  Key '{key}' in {shard_file_path.name} results in 0 rows to keep. Skipping.")
                    continue

                tensor_subset = tensor_full[:rows_for_this_key]  # Changed variable name for clarity

                # MODIFIED: Variable name updated
                current_agg_tensor = aggregated_quartered_data.get(key)
                if current_agg_tensor is None or current_agg_tensor.numel() == 0:
                    aggregated_quartered_data[key] = tensor_subset
                else:
                    try:
                        # MODIFIED: Variable names updated
                        aggregated_quartered_data[key] = torch.cat((current_agg_tensor, tensor_subset), dim=0)
                    except RuntimeError as e:
                        logging.error(
                            f"RuntimeError concatenating key '{key}' from shard {shard_file_path.name}: {e}. Agg shape: {current_agg_tensor.shape}, Subset tensor shape: {tensor_subset.shape}",
                            exc_info=True)
                        continue
            del current_shard_data, tensor_full, tensor_subset
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Major error processing shard {shard_file_path.name}: {e}", exc_info=True)
            continue

    num_total_sequences_final = 0
    # MODIFIED: Variable name updated
    if aggregated_quartered_data.get("label") is not None and aggregated_quartered_data["label"].numel() > 0:
        num_total_sequences_final = aggregated_quartered_data["label"].shape[0]

    if num_total_sequences_final == 0:
        logging.error("No data aggregated. Output file will not be created.")
        return

    for key_check in expected_keys_in_shard:
        # MODIFIED: Variable name updated
        tensor_val = aggregated_quartered_data.get(key_check)
        if tensor_val is None or tensor_val.shape[0] != num_total_sequences_final:
            logging.warning(
                f"Final check: Key '{key_check}' has {tensor_val.shape[0] if tensor_val is not None else 'None'} sequences, expected {num_total_sequences_final}. Reinitializing with zeros/ones.")
            # MODIFIED: Variable name updated
            if key_check == "packet_seq":
                aggregated_quartered_data[key_check] = torch.zeros(
                    (num_total_sequences_final, max_s_len_for_shards, num_numerical_features), dtype=torch.float32)
            elif key_check == "label":
                aggregated_quartered_data[key_check] = torch.zeros((num_total_sequences_final,),
                                                                   dtype=torch.long)
            elif key_check == "attention_mask":
                aggregated_quartered_data[key_check] = torch.ones((num_total_sequences_final, max_s_len_for_shards),
                                                                  dtype=torch.float32)
            elif key_check in categorical_columns_packets:
                aggregated_quartered_data[key_check] = torch.zeros((num_total_sequences_final, max_s_len_for_shards),
                                                                   dtype=torch.long)

    # MODIFIED: Variable name updated
    final_output_dict_to_save = {k: v for k, v in aggregated_quartered_data.items() if v is not None}

    logging.info(f"Final subset (1/4) aggregation completed in {time.time() - aggregation_start_time:.2f}s.")

    try:
        SUBSET_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_output_dict_to_save, SUBSET_OUTPUT_FILE)
        logging.info(
            f"[DONE] Merged SUBSET (1/4) dataset saved to {SUBSET_OUTPUT_FILE.resolve()} in {time.time() - aggregation_start_time:.2f}s (since aggregation start)")
    except Exception as e:
        logging.error(f"Failed to save the final subset output file: {e}", exc_info=True)
        print(f"DEBUG: FAILED TO SAVE FINAL SUBSET to {SUBSET_OUTPUT_FILE.resolve()}: {e}", flush=True)
        return

    logging.info(f"--- Subset Creation (1/4) Finished ---")


# --- Script Entry Point ---
if __name__ == '__main__':
    print(f"DEBUG: Script __main__ started. About to call setup_logging() for {LOG_FILE_PATH_OBJ.name}", flush=True)
    setup_logging()
    print(f"DEBUG: Returned from setup_logging in __main__.", flush=True)

    logging.info("--- Script execution started from __main__ ---")
    try:
        # MODIFIED: Function name updated
        create_quartered_subset()
    except Exception as e:
        print(f"DEBUG: Unhandled exception in create_quartered_subset or __main__: {e}", flush=True)
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        raise

    logging.info("--- Script execution finished from __main__ ---")
    print(f"DEBUG: Script __main__ finished.", flush=True)