# scripts/run_packet_preprocessing_subset.py
import os
import sys
import logging
from pathlib import Path

# --- Add project root to sys.path for module imports ---
# This assumes the 'scripts' directory is one level below the project root (e.g., project_root/scripts)
# Adjust if your structure is different.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from pipeline.preprocess_pipeline import run_preprocessing
    from pipeline.config import (
        categorical_columns as packet_categorical_columns, # From your original config for packets
        numerical_columns as packet_numerical_columns,   # From your original config for packets
        LABEL_MAPPING as PACKET_LABEL_MAPPING
    )
    print("✅ Successfully imported pipeline modules and packet configurations.")
except ImportError as e:
    print(f"❌ Error importing necessary modules: {e}")
    print("   Please ensure 'pipeline.preprocess_pipeline' and 'pipeline.config' are accessible.")
    print(f"   Current sys.path: {sys.path}")
    print(f"   Assumed PROJECT_ROOT: {PROJECT_ROOT}")
    exit(1)

# --- Configuration for this specific run ---

# **** USER: Define paths to your PACKET data and where outputs should go ****
PACKET_DATA_DIR          = '../../dataset/raw_packets_subset' # INPUT: Directory with raw PACKET CSVs for the subset
OUTPUT_PT_FILE           = '../../dataset/processed_packet_subset.pt' # OUTPUT: Where the final .pt file will be saved
# These files need to be generated specifically for your PACKET data features
PACKET_STATS_PATH        = 'packet_standardization_stats.npz'
PACKET_CATEGORY_MAPPINGS = 'packet_category_mappings.json'

# --- Parameters for subset processing ---
# Set TEST_MODE to True to process only a limited number of rows from each file
# This is useful for quick tests and debugging the pipeline.
# Set to False to process all rows in the files within PACKET_DATA_DIR.
TEST_MODE                = True
ROWS_PER_FILE_LIMIT      = 1000  # Max rows to read per CSV file if TEST_MODE is True (e.g., 1000, 20000)
                               # Set to None if TEST_MODE is False to process all rows.

# --- Parameters for packet sequence processing ---
MAX_SEQ_LEN              = 64    # As used in your original packet preprocessing
STREAM_ID_COLUMN_NAME    = 'stream' # The column name in your packet CSVs that identifies flows/streams

# --- Execution parameters ---
NUM_WORKERS              = os.cpu_count() // 2 or 1 # Number of parallel processes
CHECKPOINT_BASE_DIR      = "pipeline_checkpoints_packet_subset" # Base directory for checkpoint files
LOG_DIR                  = "pipeline_logs_packet_subset"      # Base directory for log files

if __name__ == "__main__":
    # Basic logging for this script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (RunPacketSubset) %(message)s')
    logging.info(f"--- Starting Packet Data Subset Preprocessing ---")
    logging.info(f"Input Packet CSVs Directory: {PACKET_DATA_DIR}")
    logging.info(f"Output .pt File: {OUTPUT_PT_FILE}")
    logging.info(f"Test Mode: {TEST_MODE}, Rows per File Limit: {ROWS_PER_FILE_LIMIT if TEST_MODE else 'All'}")

    # --- Validate essential input paths ---
    if not os.path.isdir(PACKET_DATA_DIR):
        logging.error(f"❌ Packet data directory not found: {PACKET_DATA_DIR}")
        exit(1)
    if not os.path.exists(PACKET_STATS_PATH):
        logging.error(f"❌ Packet standardization stats file not found: {PACKET_STATS_PATH}")
        logging.error(f"   Please generate this file based on your FULL packet dataset's numerical columns.")
        exit(1)
    if not os.path.exists(PACKET_CATEGORY_MAPPINGS):
        logging.error(f"❌ Packet category mappings file not found: {PACKET_CATEGORY_MAPPINGS}")
        logging.error(f"   Please generate this file based on your FULL packet dataset's categorical columns.")
        exit(1)
    if not STREAM_ID_COLUMN_NAME in packet_categorical_columns + packet_numerical_columns:
        # Stream ID might be categorical or just an identifier not in numerical/categorical lists for modeling
        # For safety, we just need it to be a column that can be loaded.
        # The io_utils.load_csv_to_df will load it if it's part of required_cols.
        # Here, we ensure it's added to the list of columns to be loaded by io_utils.
        # The packet_processor will then use it for grouping.
        logging.info(f"Note: '{STREAM_ID_COLUMN_NAME}' will be used for grouping streams.")


    # --- Construct the configuration dictionary for the pipeline ---
    packet_processing_config = {
        'dataset_dir': PACKET_DATA_DIR,
        'output_file': OUTPUT_PT_FILE,
        'data_type': 'packet', # Crucial for routing to packet_processor
        'column_names': {
            'numerical': packet_numerical_columns,
            'categorical': packet_categorical_columns,
            'stream_id_col': STREAM_ID_COLUMN_NAME
        },
        'label_mapping_dict': PACKET_LABEL_MAPPING,
        'standardization_stats_path': PACKET_STATS_PATH,
        'category_mappings_path': PACKET_CATEGORY_MAPPINGS,
        'processing_params': {
            'max_seq_len': MAX_SEQ_LEN,
            # 'default_unknown_code_cat': 0 # Example, if needed by your feature_processor
        },
        'execution_params': {
            'test_mode': TEST_MODE,
            'rows_per_file': ROWS_PER_FILE_LIMIT if TEST_MODE else None,
            'num_workers': NUM_WORKERS
        },
        'checkpoint_dir_base': CHECKPOINT_BASE_DIR,
        'log_dir': LOG_DIR
    }

    # --- Run the preprocessing pipeline ---
    try:
        logging.info("Calling run_preprocessing for packets...")
        run_preprocessing(packet_processing_config)
        logging.info(f"✅ Packet data subset preprocessing finished successfully.")
        logging.info(f"   Output saved to: {OUTPUT_PT_FILE}")
    except FileNotFoundError as fnf_error:
        logging.error(f"❌ FileNotFoundError during pipeline execution: {fnf_error}")
    except RuntimeError as rt_error:
        logging.error(f"❌ RuntimeError during pipeline execution: {rt_error}")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during packet preprocessing: {type(e).__name__} - {e}", exc_info=True)

