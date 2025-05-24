# create_subset_from_shards.py
import os
import glob
import numpy as np
# import pandas as pd # Not strictly needed here if only using config vars
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import datetime
import time
import json  # Added for loading config.json

from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING

LOG_FILE_BASENAME = "subset_aggregation"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{LOG_FILE_BASENAME}_{timestamp}.log"

SHARD_CHECKPOINT_DIR = Path("checkpoints_packet_shards")
SUBSET_OUTPUT_FILE = Path("../../../dataset/packet_subset_halved.pt")


def setup_logging():
    # Simplified for now to ensure it runs
    print("DEBUG: Inside setup_logging()", flush=True)
    try:
        logging.basicConfig(
            filename=LOG_FILE,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(console_handler)
        logging.getLogger().setLevel(logging.INFO)  # Ensure root logger level is also INFO
        print(f"DEBUG: logging.basicConfig done. Log file should be at {Path(LOG_FILE).resolve()}", flush=True)
    except Exception as e:
        print(f"DEBUG: ERROR during setup_logging: {e}", flush=True)


def create_halved_subset():
    # These prints go to STDOUT, which Slurm captures in the .out file
    print("DEBUG: Entered create_halved_subset() function.", flush=True)

    setup_logging()

    print("DEBUG: Returned from setup_logging().", flush=True)
    logging.info("--- This is a test INFO log from create_halved_subset after setup ---")  # Test logging

    max_s_len_for_shards = 64
    try:
        print("DEBUG: Attempting to load config.json...", flush=True)
        with open("config.json", 'r') as f:
            config_data = json.load(f)
            max_s_len_for_shards = config_data['model_architecture']['max_seq_len_packet']
        logging.info(f"Successfully loaded MAX_SEQ_LEN = {max_s_len_for_shards} from config.json")
        print(f"DEBUG: Loaded MAX_SEQ_LEN = {max_s_len_for_shards} from config.json", flush=True)
    except FileNotFoundError:
        logging.warning(
            f"config.json not found in CWD ({os.getcwd()}). Using default MAX_SEQ_LEN = {max_s_len_for_shards}.")
        print(f"DEBUG: config.json not found. Using default MAX_SEQ_LEN = {max_s_len_for_shards}", flush=True)
    except Exception as e:
        logging.warning(
            f"Error loading MAX_SEQ_LEN from config.json ({e}). Using default MAX_SEQ_LEN = {max_s_len_for_shards}.")
        print(f"DEBUG: Error loading config.json ({e}). Using default MAX_SEQ_LEN = {max_s_len_for_shards}", flush=True)

    logging.info(f"--- Starting Subset Creation from Halved Shards (MAX_SEQ_LEN={max_s_len_for_shards}) ---")
    print(f"DEBUG: Starting main logic of subset creation. MAX_SEQ_LEN is {max_s_len_for_shards}", flush=True)

    # ... (rest of your create_halved_subset function as I provided previously) ...
    # Ensure the rest of the function (loading shards, halving, concatenating, saving) follows here.
    # For brevity, I'm not pasting the entire shard processing loop again, but it should be here.
    # Make sure the `logging.info` and `logging.debug` calls within that loop are also present.

    all_shard_files = sorted(SHARD_CHECKPOINT_DIR.glob("*.pt"))
    total_shards_to_process = len(all_shard_files)

    if not all_shard_files:
        logging.error(f"No shard files found in {SHARD_CHECKPOINT_DIR}. Cannot create subset.")
        print(f"DEBUG: No shard files found in {SHARD_CHECKPOINT_DIR}. Exiting.", flush=True)
        return

    logging.info(f"Found {total_shards_to_process} shard files to process.")
    print(f"DEBUG: Found {total_shards_to_process} shard files.", flush=True)
    # ... (THE REST OF THE AGGREGATION LOGIC FROM THE PREVIOUS VERSION OF create_subset_from_shards.py) ...
    # This includes initializing aggregated_halved_data, looping through shards,
    # loading, halving, concatenating, and finally saving SUBSET_OUTPUT_FILE.
    # For example:
    aggregation_start_time = time.time()
    aggregated_halved_data: Dict[str, Optional[torch.Tensor]] = {
        "packet_seq": None, "label": None, "attention_mask": None,
        **{col: None for col in categorical_columns_packets}
    }
    expected_keys_in_shard = ["packet_seq", "label", "attention_mask"] + categorical_columns_packets
    num_numerical_features = len(numerical_columns_packets)

    for shard_idx, shard_file_path in enumerate(all_shard_files):
        logging.info(f"Processing shard {shard_idx + 1}/{total_shards_to_process}: {shard_file_path.name}")
        # ... (The rest of the shard processing and concatenation loop) ...
        # Remember to use max_s_len_for_shards instead of MAX_SEQ_LEN internally

    # ... (Final saving logic) ...
    logging.info(f"--- Subset Creation Finished ---")
    print("DEBUG: Subset creation function finished.", flush=True)


if __name__ == '__main__':
    print("DEBUG: Script __main__ started.", flush=True)
    # MAX_SEQ_LEN is now loaded inside create_halved_subset
    create_halved_subset()
    print("DEBUG: Script __main__ finished.", flush=True)