# create_subset_from_shards.py
import os
import glob
import numpy as np
import pandas as pd  # Not strictly needed here, but for consistency if config is shared
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import datetime
import time

# Assuming this script is in the same root as 'pipeline' and 'utils'
# or that these modules are in the Python path.
# If running from P6-Packet_FlowTransformer directory:
from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING

# If utils are also needed, e.g. for io_utils.list_csv_files, but here we glob directly

# --- Configuration ---
LOG_FILE_BASENAME = "subset_aggregation"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{LOG_FILE_BASENAME}_{timestamp}.log"

# This is where your individual processed shards are located
SHARD_CHECKPOINT_DIR = Path("checkpoints_packet_shards")

# Define the name for your new output file containing the halved data
# This will be a SINGLE file. If it's still too big, we might need to part this too.
SUBSET_OUTPUT_FILE = Path("../../../dataset/packet_subset_halved.pt")  # Adjust path as needed

# These should match the parameters used when creating the shards
# MAX_SEQ_LEN is crucial for initializing empty tensors correctly if needed.
# If your pipeline.config is accessible, you might load it here, otherwise define:
try:
    from P6_Packet_FlowTransformer.config import MAIN_CONFIG  # Example if you have a central config loader

    MAX_SEQ_LEN = MAIN_CONFIG['model_architecture']['max_seq_len_packet']
except (ImportError, KeyError):
    logging.warning("Could not load MAX_SEQ_LEN from a central config, using default 64. Ensure this is correct.")
    MAX_SEQ_LEN = 64


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",
        level=logging.INFO,  # INFO level might be better for this script
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    )
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console_handler)


def create_halved_subset():
    setup_logging()
    logging.info(f"--- Starting Subset Creation from Halved Shards ---")
    logging.info(f"Reading shards from: {SHARD_CHECKPOINT_DIR.resolve()}")
    logging.info(f"Output subset file will be: {SUBSET_OUTPUT_FILE.resolve()}")
    logging.info(f"Using MAX_SEQ_LEN: {MAX_SEQ_LEN}")

    all_shard_files = sorted(SHARD_CHECKPOINT_DIR.glob("*.pt"))
    total_shards_to_process = len(all_shard_files)

    if not all_shard_files:
        logging.error(f"No shard files found in {SHARD_CHECKPOINT_DIR}. Cannot create subset.")
        return

    logging.info(f"Found {total_shards_to_process} shard files to process.")
    aggregation_start_time = time.time()

    # Initialize aggregated_data dictionary to hold the growing concatenated HALVED tensors
    aggregated_halved_data: Dict[str, Optional[torch.Tensor]] = {
        "packet_seq": None, "label": None, "attention_mask": None,
        **{col: None for col in categorical_columns_packets}
    }
    # These are the keys expected within each shard .pt file
    expected_keys_in_shard = ["packet_seq", "label", "attention_mask"] + categorical_columns_packets
    num_numerical_features = len(numerical_columns_packets)

    for shard_idx, shard_file_path in enumerate(all_shard_files):
        logging.info(f"Processing shard {shard_idx + 1}/{total_shards_to_process}: {shard_file_path.name}")
        try:
            current_shard_data = torch.load(shard_file_path, map_location='cpu')  # Load to CPU

            # Determine the number of rows to keep (half) from the 'label' tensor
            if "label" not in current_shard_data or not isinstance(current_shard_data["label"], torch.Tensor) or \
                    current_shard_data["label"].numel() == 0:
                logging.warning(f"Shard {shard_file_path.name} has no 'label' data or it's empty. Skipping this shard.")
                continue

            num_rows_in_shard = current_shard_data["label"].shape[0]
            if num_rows_in_shard == 0:
                logging.info(f"Shard {shard_file_path.name} contains 0 sequences. Skipping.")
                continue

            num_rows_to_keep = num_rows_in_shard // 2
            if num_rows_to_keep == 0 and num_rows_in_shard > 0:  # Ensure at least 1 row if original has rows
                num_rows_to_keep = 1

            logging.debug(
                f"  Shard {shard_file_path.name}: total rows {num_rows_in_shard}, keeping first {num_rows_to_keep} rows.")

            for key in expected_keys_in_shard:
                if key not in current_shard_data:
                    logging.warning(f"Key '{key}' not found in shard {shard_file_path.name}.")
                    # Initialize if it's the first time seeing this key in aggregated_halved_data
                    if aggregated_halved_data.get(key) is None:
                        if key == "packet_seq":
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN, num_numerical_features),
                                                                      dtype=torch.float32)
                        elif key == "label":
                            aggregated_halved_data[key] = torch.empty((0,), dtype=torch.long)
                        elif key == "attention_mask":
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN), dtype=torch.float32)
                        elif key in categorical_columns_packets:
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN), dtype=torch.long)
                    continue

                tensor_full = current_shard_data[key]
                if not isinstance(tensor_full, torch.Tensor):
                    logging.warning(
                        f"Data for key '{key}' in shard {shard_file_path.name} is not a tensor. Skipping this key for this shard.")
                    continue

                if tensor_full.numel() == 0 or tensor_full.shape[0] == 0:  # If this specific tensor is empty
                    logging.debug(f"Key '{key}' in shard {shard_file_path.name} is an empty tensor. Skipping.")
                    if aggregated_halved_data.get(key) is None:  # Initialize if not already
                        if key == "packet_seq":
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN, num_numerical_features),
                                                                      dtype=torch.float32)
                        elif key == "label":
                            aggregated_halved_data[key] = torch.empty((0,), dtype=torch.long)
                        elif key == "attention_mask":
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN), dtype=torch.float32)
                        elif key in categorical_columns_packets:
                            aggregated_halved_data[key] = torch.empty((0, MAX_SEQ_LEN), dtype=torch.long)
                    continue

                # Slice the tensor to get the first half of the sequences
                if tensor_full.shape[0] < num_rows_to_keep:  # Should not happen if num_rows_to_keep is based on label
                    logging.warning(
                        f"  Tensor for key '{key}' in {shard_file_path.name} has fewer rows ({tensor_full.shape[0]}) than num_rows_to_keep ({num_rows_to_keep}). Taking all rows.")
                    tensor_half = tensor_full
                else:
                    tensor_half = tensor_full[:num_rows_to_keep]

                if tensor_half.numel() == 0:  # If after slicing it's empty (e.g. num_rows_to_keep was 0)
                    continue

                # Concatenate
                current_agg_tensor = aggregated_halved_data.get(key)
                if current_agg_tensor is None or current_agg_tensor.numel() == 0:
                    aggregated_halved_data[key] = tensor_half
                else:
                    try:
                        aggregated_halved_data[key] = torch.cat((current_agg_tensor, tensor_half), dim=0)
                    except RuntimeError as e:
                        logging.error(
                            f"RuntimeError concatenating key '{key}' from shard {shard_file_path.name}: {e}. Aggregated shape: {current_agg_tensor.shape}, Shard tensor (halved) shape: {tensor_half.shape}",
                            exc_info=True)
                        continue  # Try to continue with other keys/shards

            del current_shard_data, tensor_full, tensor_half  # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except FileNotFoundError:
            logging.error(f"Shard file {shard_file_path.name} not found during aggregation. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error processing shard {shard_file_path.name} during aggregation: {e}", exc_info=True)
            continue

            # Final integrity check for the aggregated data
    num_total_sequences_final = 0
    if aggregated_halved_data.get("label") is not None and aggregated_halved_data["label"].numel() > 0:
        num_total_sequences_final = aggregated_halved_data["label"].shape[0]

    if num_total_sequences_final == 0:
        logging.error(
            "No data aggregated after processing all shards. Output file will not be created or will be empty.")
        # Optionally save an empty structure if downstream expects a file
        # For now, just returning.
        return

    for key_check in expected_keys_in_shard:
        tensor_check = aggregated_halved_data.get(key_check)
        if tensor_check is None or tensor_check.numel() == 0 or tensor_check.shape[0] != num_total_sequences_final:
            logging.warning(
                f"Final aggregated data for key '{key_check}' is missing, empty, or has inconsistent sequence count ({tensor_check.shape[0] if tensor_check is not None else 'None'}) vs labels ({num_total_sequences_final}). Initializing appropriately.")
            if key_check == "packet_seq":
                aggregated_halved_data[key_check] = torch.zeros(
                    (num_total_sequences_final, MAX_SEQ_LEN, num_numerical_features), dtype=torch.float32)
            elif key_check == "label":
                aggregated_halved_data[key_check] = torch.zeros((num_total_sequences_final,),
                                                                dtype=torch.long)  # Should not happen if num_total_sequences_final > 0
            elif key_check == "attention_mask":
                aggregated_halved_data[key_check] = torch.ones((num_total_sequences_final, MAX_SEQ_LEN),
                                                               dtype=torch.float32)  # Default to ones
            elif key_check in categorical_columns_packets:
                aggregated_halved_data[key_check] = torch.zeros((num_total_sequences_final, MAX_SEQ_LEN),
                                                                dtype=torch.long)

    final_output_dict_to_save = {k: v for k, v in aggregated_halved_data.items() if v is not None}

    logging.info(f"Final subset aggregation completed in {time.time() - aggregation_start_time:.2f}s.")

    SUBSET_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    save_output_start_time = time.time()
    torch.save(final_output_dict_to_save, SUBSET_OUTPUT_FILE)
    logging.info(
        f"[DONE] Merged SUBSET dataset from halved shards saved to {SUBSET_OUTPUT_FILE} in {time.time() - save_output_start_time:.2f}s")

    final_label_tensor = final_output_dict_to_save.get("label")
    if final_label_tensor is not None and final_label_tensor.numel() > 0:
        logging.info(f"   Total sequences in this subset output file: {final_label_tensor.shape[0]}")
        try:
            label_counts_np_final = np.bincount(final_label_tensor.cpu().numpy())
            logging.info("Final Label distribution in this subset dataset:")
            for lbl_id_report_final, cnt_report_final in enumerate(label_counts_np_final):
                if cnt_report_final > 0:
                    label_name_report_final = next((name_report for name_report, val_report in LABEL_MAPPING.items() if
                                                    val_report == lbl_id_report_final), "Unknown")
                    logging.info(
                        f"  Class '{label_name_report_final}' ({lbl_id_report_final}): {cnt_report_final} sequences")
        except Exception as e:
            logging.error(f"Error reporting label distribution for subset: {e}")
    else:
        logging.warning("No 'label' data in final subset output to report distribution.")

    logging.info(f"--- Subset Creation Finished ---")


if __name__ == '__main__':
    # Make sure numerical_columns_packets and categorical_columns_packets are available
    # This might require importing them directly or loading from a shared config if this script
    # is moved outside the pipeline directory.
    # For simplicity, assuming pipeline.config is accessible as in process.py

    # Example of how MAX_SEQ_LEN might be obtained if not hardcoding:
    # This requires your config.json to be accessible and structured as expected.
    try:
        import json

        with open("config.json", 'r') as f:  # Assuming config.json is in the same dir or adjust path
            config_data = json.load(f)
            MAX_SEQ_LEN = config_data['model_architecture']['max_seq_len_packet']
            logging.info(f"Loaded MAX_SEQ_LEN = {MAX_SEQ_LEN} from config.json")
    except Exception as e:
        logging.warning(
            f"Could not load MAX_SEQ_LEN from config.json ({e}), using default 64. Ensure this is correct for your shards.")
        MAX_SEQ_LEN = 64  # Fallback, ensure this matches shard creation

    create_halved_subset()