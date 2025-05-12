# pipeline/preprocess_pipeline.py
import os
import pandas as pd
import numpy as np
import torch
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
import datetime
import time

# --- Import project-specific utility and processor modules ---
# Assuming they are in utils/ and pipeline/ relative to where this might be called from,
# or that the project structure is set up in PYTHONPATH.
try:
    from utils import io_utils, label_utils, category_mapping_utils
    from pipeline import feature_processor, packet_processor, flow_processor

    print("✅ Successfully imported utility and processor modules.")
except ImportError as e:
    print(f"❌ Error importing modules: {e}. Ensure PYTHONPATH is set correctly or files are in expected locations.")
    # Depending on execution context, you might need to adjust import paths:
    # e.g., from ..utils import io_utils (if this is deeper in pipeline)
    raise


# Configure basic logging for this module
# The main script calling run_preprocessing should ideally set up global logging.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (preprocess_pipeline) %(message)s')


def _process_single_file_shard(
        file_path: str,
        config: Dict[str, Any]  # Contains all necessary configurations
) -> Optional[Tuple[str, Dict[str, np.ndarray]]]:
    """
    Processes a single input CSV file into a shard of processed NumPy arrays.
    This function is designed to be run in a parallel worker.

    Args:
        file_path: Path to the input CSV file.
        config: A comprehensive configuration dictionary. Expected keys:
            'data_type': 'packet' or 'flow'
            'column_names': {'numerical': List[str], 'categorical': List[str], 'stream_id_col' (if packet): str}
            'label_mapping_dict': Dict[str, int]
            'standardization_stats': Loaded stats from feature_processor.load_standardization_stats
            'category_mappings': Loaded mappings from category_mapping_utils.load_mappings
            'processing_params': {'max_seq_len' (if packet): int, 'default_unknown_code_cat': int}
            'execution_params': {'test_mode': bool, 'rows_per_file': int | None}

    Returns:
        A tuple (file_path, shard_data_dict) where shard_data_dict contains
        NumPy arrays for the processed data from this file.
        Returns None if processing fails for this file.
    """
    try:
        # logging.info(f"Processing shard for file: {Path(file_path).name}") # Can be too verbose in parallel

        # 1. Load CSV to DataFrame
        # Determine required columns based on data_type for efficient loading
        base_required_cols = config['column_names']['numerical'] + config['column_names']['categorical']
        if config['data_type'] == 'packet':
            stream_col = config['column_names'].get('stream_id_col')
            if not stream_col:
                logging.error(
                    f"Missing 'stream_id_col' in column_names config for packet data processing of {file_path}.")
                return file_path, None  # Return file_path to mark as attempted
            base_required_cols.append(stream_col)

        load_result = io_utils.load_csv_to_df(
            file_path,
            required_cols=list(set(base_required_cols)),  # Ensure unique columns
            test_mode=config['execution_params']['test_mode'],
            rows_per_file=config['execution_params']['rows_per_file'],
            low_memory=False  # Usually better for feature processing
        )
        if load_result is None:
            logging.warning(f"Failed to load df from {file_path}. Skipping shard.")
            return file_path, None
        df, _ = load_result

        # 2. Find Label
        label = label_utils.find_label_from_path(file_path, config['label_mapping_dict'])
        if label == -1:
            logging.warning(f"No label found for {file_path}. Skipping shard.")
            return file_path, None

        # 3. Process Numerical Features
        df = feature_processor.process_numerical_features(
            df,
            config['column_names']['numerical'],
            config['standardization_stats']
        )

        # 4. Encode Categorical Features
        # Ensure NaNs in categorical columns are filled with "unknown" string *before* encoding
        # if that's the desired strategy for unknown handling.
        # The `encode_categorical_features` expects string inputs for mapping.
        for cat_col in config['column_names']['categorical']:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].fillna("unknown").astype(str)
            else:
                logging.warning(f"Categorical column '{cat_col}' specified in config not found in df from {file_path}")

        df = feature_processor.encode_categorical_features(
            df,
            config['column_names']['categorical'],
            config['category_mappings'],
            # default_unknown_code=config['processing_params']['default_unknown_code_cat'] # Pass if needed
        )

        # 5. Data-Type Specific Processing (Packet Sequencing or Flow Vector Prep)
        shard_data_dict: Dict[str, np.ndarray] = {}

        if config['data_type'] == 'packet':
            packet_processing_result = packet_processor.create_packet_sequences_from_processed_df(
                processed_df=df,
                file_label=label,  # Pass the single label for all sequences from this file
                numerical_cols=config['column_names']['numerical'],
                categorical_cols=config['column_names']['categorical'],
                stream_col_name=config['column_names']['stream_id_col'],
                max_seq_len=config['processing_params']['max_seq_len'],
                category_mappings=config['category_mappings']  # For unknown padding codes
            )
            if packet_processing_result is None:
                logging.warning(f"Packet sequence creation failed for {file_path}. Skipping shard.")
                return file_path, None

            num_seqs, labels_list, masks, cat_seqs_dict = packet_processing_result

            if not num_seqs:  # No sequences generated
                logging.info(f"No sequences generated from {file_path}. Skipping shard.")
                return file_path, None

            shard_data_dict['packet_seq_np'] = np.stack(num_seqs) if num_seqs else np.array([])
            shard_data_dict['label_np'] = np.array(labels_list, dtype=np.int64) if labels_list else np.array([])
            shard_data_dict['attention_mask_np'] = np.stack(masks) if masks else np.array([])
            for col_name, seq_list in cat_seqs_dict.items():
                shard_data_dict[f"{col_name}_np"] = np.stack(seq_list) if seq_list else np.array([])

        elif config['data_type'] == 'flow':
            # For flows, each row in df is a sample. Labels list needs to match df length.
            flow_labels = [label] * len(df)  # Assign file_label to all flows in this df
            flow_processing_result = flow_processor.prepare_flow_data_for_tensor_conversion(
                processed_df=df,
                labels=flow_labels,
                numerical_cols=config['column_names']['numerical'],
                categorical_cols=config['column_names']['categorical']
            )
            if flow_processing_result is None:
                logging.warning(f"Flow data preparation failed for {file_path}. Skipping shard.")
                return file_path, None

            num_vectors, cat_vectors, label_vectors = flow_processing_result
            shard_data_dict['numerical_features_np'] = num_vectors
            shard_data_dict['categorical_features_np'] = cat_vectors
            shard_data_dict['label_np'] = label_vectors
        else:
            logging.error(f"Unknown data_type '{config['data_type']}' for {file_path}.")
            return file_path, None

        # logging.info(f"Successfully processed shard for: {Path(file_path).name}")
        return file_path, shard_data_dict

    except Exception as e:
        logging.error(f"Unhandled error processing file {file_path}: {type(e).__name__} - {e}", exc_info=True)
        return file_path, None


def run_preprocessing(config: Dict[str, Any]) -> None:
    """
    Main orchestration function for preprocessing either packet or flow data.

    Args:
        config: A comprehensive configuration dictionary. Expected keys:
            'dataset_dir': str
            'output_file': str
            'data_type': str ('packet' or 'flow')
            'column_names': {'numerical': List[str], 'categorical': List[str], 'stream_id_col' (if packet): str}
            'label_mapping_dict': Dict[str, int] (The actual mapping, not path)
            'standardization_stats_path': str
            'category_mappings_path': str
            'processing_params': {'max_seq_len' (if packet): int, 'default_unknown_code_cat' (optional): int}
            'execution_params': {'test_mode': bool, 'rows_per_file': int | None, 'num_workers': int}
            'checkpoint_dir_base': str (e.g., "checkpoints")
    """
    start_time_pipeline = time.time()

    # --- Setup Logging and Checkpoint Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Main log file for the pipeline run
    pipeline_log_file = Path(config.get("log_dir", ".")) / f"preprocess_pipeline_{config['data_type']}_{timestamp}.log"
    pipeline_log_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers and add new one for this run
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=str(pipeline_log_file),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(processName)s-%(threadName)s] [%(levelname)s] (%(module)s) %(message)s",
    )
    logging.info(f"🚀 Starting Preprocessing Pipeline for data_type: '{config['data_type']}' 🚀")
    logging.info(f"Full configuration: {config}")

    checkpoint_dir = Path(config['checkpoint_dir_base']) / f"{config['data_type']}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using checkpoint directory: {checkpoint_dir}")

    # --- Load External Resources ---
    logging.info("Loading external resources (stats, mappings)...")
    standardization_stats = feature_processor.load_standardization_stats(config['standardization_stats_path'])
    if standardization_stats is None:
        logging.error("Failed to load standardization stats. Exiting.")
        return

    category_mappings = category_mapping_utils.load_mappings(config['category_mappings_path'])
    if category_mappings is None:  # load_mappings raises error, so this might not be hit
        logging.error("Failed to load category mappings. Exiting.")
        return

    # Augment config with loaded resources for worker processes
    worker_config = config.copy()
    worker_config['standardization_stats'] = standardization_stats
    worker_config['category_mappings'] = category_mappings
    # label_mapping_dict is already assumed to be in config

    # --- List Input Files and Determine Pending Work ---
    all_input_csv_files = io_utils.list_csv_files(config['dataset_dir'], recursive=True)
    if not all_input_csv_files:
        logging.error(f"No CSV files found in dataset_dir: {config['dataset_dir']}. Exiting.")
        return
    logging.info(f"Found {len(all_input_csv_files)} total CSV files in {config['dataset_dir']}.")

    pending_files_to_process: List[str] = []
    for f_path in all_input_csv_files:
        shard_filename = Path(f_path).stem + f"_{config['data_type']}_shard.pt"
        if not (checkpoint_dir / shard_filename).exists():
            pending_files_to_process.append(f_path)

    num_already_checkpointed = len(all_input_csv_files) - len(pending_files_to_process)
    if num_already_checkpointed > 0:
        logging.info(f"{num_already_checkpointed} files already have processed shards in checkpoint directory.")
    logging.info(f"{len(pending_files_to_process)} files pending full processing.")

    if not pending_files_to_process and num_already_checkpointed == 0:  # No files at all
        logging.error("No files to process and no checkpoints found. Exiting.")
        return
    elif not pending_files_to_process and num_already_checkpointed > 0:
        logging.info("All files seem to be processed (checkpoints exist). Proceeding to aggregation.")

    # --- Parallel Processing of Shards ---
    num_workers = config['execution_params'].get('num_workers', os.cpu_count())
    logging.info(f"Starting parallel shard processing with {num_workers} workers...")

    # Using ProcessPoolExecutor as feature processing can be CPU intensive
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Pass the augmented worker_config to each worker
        futures = [
            executor.submit(_process_single_file_shard, file_path, worker_config)
            for file_path in pending_files_to_process
        ]

        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            try:
                original_file_path, shard_data_dict = future.result()
                if shard_data_dict is not None and all(arr.size > 0 for arr in shard_data_dict.values() if
                                                       isinstance(arr, np.ndarray)):  # Check if data is not empty
                    shard_filename = Path(original_file_path).stem + f"_{config['data_type']}_shard.pt"
                    torch.save(shard_data_dict, checkpoint_dir / shard_filename)
                    logging.info(
                        f"[CKPT {processed_count}/{len(pending_files_to_process)}] Saved shard for {Path(original_file_path).name} to {shard_filename}")
                elif shard_data_dict is not None:  # Data was empty
                    logging.warning(
                        f"[CKPT {processed_count}/{len(pending_files_to_process)}] Processed shard for {Path(original_file_path).name} resulted in empty data. Not saved.")
                else:  # Processing failed for this file
                    logging.error(
                        f"[CKPT {processed_count}/{len(pending_files_to_process)}] Failed to process shard for {Path(original_file_path).name}.")
            except Exception as e:
                logging.error(f"Error processing a future result: {e}", exc_info=True)

    logging.info("Parallel shard processing complete.")

    # --- Aggregate Shards ---
    logging.info("Aggregating processed shards from checkpoint directory...")
    all_shard_files = sorted(checkpoint_dir.glob(f"*_{config['data_type']}_shard.pt"))
    if not all_shard_files:
        logging.error("No processed shards found in checkpoint directory. Cannot create final output.")
        return

    # Initialize lists to hold tensors from all shards
    # The keys will depend on data_type
    aggregated_data: Dict[str, List[torch.Tensor]] = defaultdict(list)
    first_shard_keys = None  # To ensure all shards have same structure

    for shard_file_path in all_shard_files:
        try:
            shard_data = torch.load(shard_file_path)  # This loads dict of NumPy arrays
            if first_shard_keys is None:
                first_shard_keys = set(shard_data.keys())
            elif set(shard_data.keys()) != first_shard_keys:
                logging.warning(f"Shard {shard_file_path.name} has different keys than first shard. Skipping.")
                continue

            for key, numpy_array in shard_data.items():
                if numpy_array.size > 0:  # Only append if not empty
                    aggregated_data[key].append(torch.from_numpy(numpy_array))
                # else: # Log if a specific array within a shard is empty
                #    logging.debug(f"Empty array for key '{key}' in shard {shard_file_path.name}")

        except Exception as e:
            logging.error(f"Error loading or processing shard {shard_file_path}: {e}")
            continue

    if not aggregated_data or not any(aggregated_data.values()):
        logging.error("Aggregation resulted in no data. Check processed shards.")
        return

    # Concatenate tensors
    final_output_dict: Dict[str, torch.Tensor] = {}
    for key, list_of_tensors in aggregated_data.items():
        if list_of_tensors:
            try:
                final_output_dict[key.replace('_np', '')] = torch.cat(list_of_tensors, dim=0)  # Remove _np suffix
                logging.info(f"Aggregated '{key}': final shape {final_output_dict[key.replace('_np', '')].shape}")
            except Exception as e:
                logging.error(f"Error concatenating tensors for key '{key}': {e}")
                # Log shapes of tensors in list_of_tensors for debugging
                # for i, t in enumerate(list_of_tensors): logging.debug(f"  Tensor {i} shape for key {key}: {t.shape}")
                return  # Stop if aggregation fails
        else:
            logging.warning(f"No tensors to concatenate for key '{key}'.")

    # --- Add Metadata ---
    final_output_dict['metadata'] = {
        'data_type': config['data_type'],
        'numerical_columns': config['column_names']['numerical'],
        'categorical_columns': config['column_names']['categorical'],
        # It's good practice to save the actual category mappings used,
        # or at least the cardinalities if the model needs them.
        # For now, let's assume the model will get cardinalities from the loaded mappings separately.
        'source_dataset_dir': config['dataset_dir'],
        'creation_timestamp': timestamp,
    }
    if config['data_type'] == 'packet':
        final_output_dict['metadata']['stream_id_col'] = config['column_names'].get('stream_id_col')
        final_output_dict['metadata']['max_seq_len'] = config['processing_params'].get('max_seq_len')

    # --- Save Final Output ---
    try:
        Path(config['output_file']).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_output_dict, config['output_file'])
        logging.info(f"✅ Successfully saved aggregated dataset to: {config['output_file']}")
    except Exception as e:
        logging.error(f"Error saving final aggregated dataset: {e}")
        return

    # --- Log Summary ---
    if 'label' in final_output_dict and final_output_dict['label'].numel() > 0:
        label_counts = np.bincount(final_output_dict['label'].numpy())
        logging.info("--- Final Output Label Distribution ---")
        for lbl_id, cnt in enumerate(label_counts):
            label_name = "Unknown_Label"
            for name, val in config['label_mapping_dict'].items():
                if val == lbl_id:
                    label_name = name
                    break
            logging.info(f"  Class '{label_name}' ({lbl_id}): {cnt} samples/sequences")
        logging.info(f"Total samples/sequences in final output: {final_output_dict['label'].shape[0]}")
    else:
        logging.warning("No 'label' key in final output or label tensor is empty. Cannot report distribution.")

    end_time_pipeline = time.time()
    logging.info(
        f"🏁 Preprocessing Pipeline Finished. Total time: {end_time_pipeline - start_time_pipeline:.2f} seconds. 🏁")


# Example of how to call this (would be in a separate script like scripts/preprocess_flows.py)
if __name__ == '__main__':
    # This block is for demonstration and testing of run_preprocessing
    # You would create separate scripts (e.g., preprocess_my_flows.py, preprocess_my_packets.py)
    # that define their specific configs and call run_preprocessing.

    # --- Dummy LABEL_MAPPING for testing ---
    TEST_LABEL_MAPPING = {
        "BenignTraffic": 0,
        "AttackTypeA": 1,
    }
    # --- Create dummy CSV files for testing ---
    test_data_dir = Path("temp_pipeline_test_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Dummy Flow Config
    flow_config_for_test = {
        'dataset_dir': str(test_data_dir / "flows"),
        'output_file': str(test_data_dir / "processed_flows_test.pt"),
        'data_type': 'flow',
        'column_names': {
            'numerical': ['FlowDuration', 'TotalPackets'],
            'categorical': ['ProtocolType', 'DeviceCategory'],
        },
        'label_mapping_dict': TEST_LABEL_MAPPING,
        'standardization_stats_path': "temp_flow_stats.npz",  # Needs to be created
        'category_mappings_path': "temp_flow_mappings.json",  # Needs to be created
        'processing_params': {},  # No special params for flows here
        'execution_params': {'test_mode': False, 'rows_per_file': None, 'num_workers': 2},
        'checkpoint_dir_base': str(test_data_dir / "checkpoints"),
        'log_dir': str(test_data_dir / "logs")
    }
    (test_data_dir / "flows").mkdir(exist_ok=True)
    pd.DataFrame({
        'FlowDuration': np.random.rand(100) * 1000, 'TotalPackets': np.random.randint(1, 50, 100),
        'ProtocolType': np.random.choice(['TCP', 'UDP', 'ICMP'], 100),
        'DeviceCategory': np.random.choice(['Camera', 'Sensor', 'BenignTraffic'], 100),  # Label in folder name
        'SomeOtherCol': range(100)
    }).to_csv(test_data_dir / "flows" / "BenignTraffic_file1.csv", index=False)
    pd.DataFrame({
        'FlowDuration': np.random.rand(80) * 2000, 'TotalPackets': np.random.randint(10, 100, 80),
        'ProtocolType': np.random.choice(['TCP', 'UDP'], 80),
        'DeviceCategory': np.random.choice(['Light', 'AttackTypeA', 'Thermostat'], 80),
    }).to_csv(test_data_dir / "flows" / "AttackTypeA_file2.csv", index=False)

    # Create dummy flow stats
    flow_num_cols = flow_config_for_test['column_names']['numerical']
    np.savez(flow_config_for_test['standardization_stats_path'],
             cols=np.array(flow_num_cols),
             mean=np.array([500, 25]), std=np.array([200, 10]),
             median=np.array([450, 22]), clip_low=np.array([0, 1]), clip_high=np.array([5000, 200]))
    # Create dummy flow mappings
    flow_cat_cols = flow_config_for_test['column_names']['categorical']
    dummy_flow_mappings_json = {
        'ProtocolType': {repr('TCP'): 0, repr('UDP'): 1, repr('ICMP'): 2, repr('unknown'): 3},
        'DeviceCategory': {repr('Camera'): 0, repr('Sensor'): 1, repr('Light'): 2, repr('Thermostat'): 3,
                           repr('BenignTraffic'): 4, repr('AttackTypeA'): 5, repr('unknown'): 6}
    }
    with open(flow_config_for_test['category_mappings_path'], 'w') as f_map:
        json.dump(dummy_flow_mappings_json, f_map)

    print("\n--- TESTING FLOW PREPROCESSING PIPELINE ---")
    try:
        run_preprocessing(flow_config_for_test)
        # Check output file
        if os.path.exists(flow_config_for_test['output_file']):
            print(f"✅ Flow processing test seemed to complete. Output: {flow_config_for_test['output_file']}")
            # loaded_data = torch.load(flow_config_for_test['output_file'])
            # print("   Keys in output:", loaded_data.keys())
            # print("   Numerical shape:", loaded_data['numerical_features'].shape)
            # print("   Categorical shape:", loaded_data['categorical_features'].shape)
            # print("   Label shape:", loaded_data['label'].shape)
    except Exception as e_test:
        print(f"❌ Test run for flow preprocessing failed: {e_test}")
        logging.exception("Test run exception:")

    # --- Dummy Packet Config (More involved due to sequences) ---
    # You would create a similar config for 'packet' data type,
    # ensuring 'stream_id_col' and 'max_seq_len' are provided.
    # And that standardization_stats_path & category_mappings_path point to
    # files appropriate for packet data.

    # print("\n(To clean up, manually delete temp_pipeline_test_data directory)")

