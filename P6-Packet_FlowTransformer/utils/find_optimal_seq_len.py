import os
import glob
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
# Assuming io_utils is in the same directory or package structure allows this import
from . import io_utils # Or adjust based on your structure e.g., import io_utils

# Configure logging if not already done
# Ensure logging is configured *before* any logging calls if running as a script
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Define the worker function at the top level ---
def _get_lengths_from_file(args: Tuple[Optional[pd.DataFrame], str]) -> List[int]:
    """
    Helper function to extract flow lengths from a single DataFrame.
    Designed to be run in a separate process.
    """
    df, file_path = args
    flow_lengths = []
    if df is not None and not df.empty:
        if "stream" in df.columns:
            # Group by 'stream' and calculate the size (length) of each group
            try:
                # Using observed=True can sometimes be faster and avoid warnings
                # depending on pandas version and data types.
                # Also explicitly handle potential non-numeric group keys if necessary.
                lengths = df.groupby("stream", sort=False, observed=True).size()
                flow_lengths.extend(lengths.tolist())
            except Exception as e:
                # Log the specific file path with the error
                logging.error(f"Error processing flows in {file_path}: {e}")
        else:
            logging.warning(f"'stream' column not found in {file_path}. Cannot calculate flow lengths.")
    elif df is None:
        logging.warning(f"Received None DataFrame for path {file_path}. Skipping.")
    # else: df is empty, already logged during loading usually or handled implicitly

    return flow_lengths
# --- End of top-level worker function ---


def calculate_flow_length_stats(
    dataset_dir: str,
    test_mode: bool = False,
    rows_per_file: Optional[int] = None,
    percentiles: List[int] = [50, 75, 90, 95, 99, 100]
) -> Dict[str, float]:
    """
    Analyzes CSV files in a directory to find the distribution of flow lengths.

    Args:
        dataset_dir: Path to the directory containing CSV files.
        test_mode: If True, load only a small subset of rows for quick testing.
        rows_per_file: Maximum rows to load per file (None for all rows).
                       Useful for large files if test_mode is False.
        percentiles: A list of percentiles to calculate for flow lengths.

    Returns:
        A dictionary containing statistics (count, mean, std, min, max,
        and specified percentiles) of the flow lengths across the dataset.
        Returns an empty dictionary if no valid flows are found.
    """
    all_files = io_utils.list_csv_files(dataset_dir)
    if not all_files:
        logging.warning(f"No CSV files found in {dataset_dir}")
        return {}

    logging.info(f"Analyzing flow lengths in {len(all_files)} files from {dataset_dir}...")

    all_flow_lengths: List[int] = []

    # Load files in parallel using ThreadPoolExecutor (good for I/O bound tasks)
    loaded_files_args = []
    # Use slightly fewer workers than CPUs for loading if IO is the bottleneck,
    # or leave as os.cpu_count() if unsure.
    num_load_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 4 # Example adjustment
    logging.info(f"Using {num_load_workers} workers for loading files.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
        # Submit loading tasks
        futures_load = {
            tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file): fp
            for fp in all_files
        }
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures_load):
            file_path = futures_load[future] # Get path for context
            try:
                result = future.result() # result is (df, file_path) or None
                if result is not None:
                    loaded_files_args.append(result)
                # else: load_csv_file should have logged the error if result is None
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")


    if not loaded_files_args:
        logging.warning("No data successfully loaded from any files.")
        return {}

    logging.info(f"Loaded {len(loaded_files_args)} files. Calculating flow lengths...")

    # Process loaded dataframes in parallel using ProcessPoolExecutor for CPU-bound task
    # Determine number of workers, avoid using too many if memory is constrained
    num_process_workers = os.cpu_count() # Use all available CPUs
    logging.info(f"Using {num_process_workers} workers for processing lengths.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_process_workers) as ppool:
        # Submit processing tasks using the top-level function
        futures_process = {
            # Pass the tuple (df, file_path) directly to the top-level function
            ppool.submit(_get_lengths_from_file, args): args[1] # Map future to file_path for logging
            for args in loaded_files_args
        }
        # Collect results
        processed_count = 0
        total_to_process = len(loaded_files_args)
        for future in concurrent.futures.as_completed(futures_process):
            file_path_processed = futures_process[future] # Get path for context
            try:
                lengths = future.result() # This is the List[int] from _get_lengths_from_file
                all_flow_lengths.extend(lengths)
                processed_count += 1
                # Log progress periodically
                if processed_count % 20 == 0 or processed_count == total_to_process:
                     logging.info(f"Processed {processed_count}/{total_to_process} files for lengths...")
            except Exception as e:
                # This catches errors happening *during* the execution of _get_lengths_from_file
                # within the worker process, or issues during result pickling/unpickling.
                logging.error(f"Error processing result for file associated with {file_path_processed}: {e}")


    if not all_flow_lengths:
        logging.warning("No valid flows found across all processed files.")
        return {}

    # Calculate statistics using numpy
    logging.info(f"Calculating final statistics for {len(all_flow_lengths)} flows...")
    flow_lengths_np = np.array(all_flow_lengths, dtype=np.int64) # Specify dtype for safety

    # Handle case where flow_lengths_np might still be empty after processing
    if flow_lengths_np.size == 0:
        logging.warning("Flow lengths array is empty after processing. No statistics calculated.")
        return {}

    stats = {
        "count": len(flow_lengths_np),
        "mean": np.mean(flow_lengths_np),
        "std_dev": np.std(flow_lengths_np),
        "min": np.min(flow_lengths_np),
        "max": np.max(flow_lengths_np),
    }

    # Calculate requested percentiles
    for p in percentiles:
        try:
            stats[f"{p}th_percentile"] = np.percentile(flow_lengths_np, p)
        except IndexError:
            logging.warning(f"Could not calculate {p}th percentile, possibly due to empty data.")
            stats[f"{p}th_percentile"] = np.nan # Or handle as appropriate

    logging.info("Flow length analysis complete.")
    # Format stats for logging - Ensure all values are handled
    formatted_stats = {
        k: f'{v:.2f}' if isinstance(v, (float, np.number)) and not np.isnan(v) else
           ('NaN' if isinstance(v, float) and np.isnan(v) else v)
        for k, v in stats.items()
    }
    logging.info(f"Stats calculated: {formatted_stats}")

    return stats

# Example Usage:
if __name__ == "__main__":
    # Configure logging for the example script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", # Added logger name
        handlers=[logging.StreamHandler()] # Log to console
    )
    # Define logger for this specific module/script
    logger = logging.getLogger(__name__) # Get logger for current module

    # --- IMPORTANT ---
    # Adjust this path relative to where you RUN the script from,
    # or use an absolute path.
    # If utils/find_optimal_seq_len.py is run from the project root,
    # the path might be "dataset/raw_dataset"
    # If run from inside the 'utils' directory, it would be "../dataset/raw_dataset"
    # The log shows '../../dataset/raw_dataset', suggesting it was run from utils/something/
    DATASET_DIRECTORY = "../../dataset/raw_dataset" # Adjust as needed!

    # Check if directory exists
    if not os.path.isdir(DATASET_DIRECTORY):
       logger.error(f"Dataset directory not found: {os.path.abspath(DATASET_DIRECTORY)}")
       logger.error("Please ensure the DATASET_DIRECTORY path is correct relative to the script execution location.")
    else:
        logger.info(f"Attempting to analyze dataset at: {os.path.abspath(DATASET_DIRECTORY)}")
        # Assume io_utils is correctly imported relative to this file's location
        # If io_utils is in the *same* directory ('utils'), use:
        # import io_utils
        # If io_utils is one level up (e.g., project root), you might need path adjustments or:
        # import sys
        # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        # import io_utils
        try:
            # Make sure the io_utils module is loaded correctly before calling this
            # This assumes the relative import `. import io_utils` works based on your structure
            # If running as a script `python -m utils.find_optimal_seq_len`, relative imports should work.
            # If running `python utils/find_optimal_seq_len.py`, they might fail.
            # You might need to change the import to `import io_utils` and ensure utils is runnable.

            # Mock io_utils if it's not available for testing the structure
            # class MockIoUtils:
            #     def list_csv_files(self, path): return []
            #     def load_csv_file(self, fp, test_mode, rows_per_file): return (None, fp)
            # uncomment below and comment the real import if io_utils is the issue
            # io_utils = MockIoUtils()

            flow_stats = calculate_flow_length_stats(DATASET_DIRECTORY, test_mode=False)

            if flow_stats:
                print("\n--- Flow Length Statistics ---")
                for key, value in flow_stats.items():
                    # Check for NaN before formatting
                    if isinstance(value, float) and np.isnan(value):
                        print(f"{key}: NaN")
                    else:
                       # Attempt to format as float, fallback to string if not possible
                        try:
                            # Format numbers (int/float) appropriately
                            if isinstance(value, (int, float, np.number)):
                                print(f"{key}: {value:.2f}")
                            else:
                                print(f"{key}: {value}") # Print non-numeric types as is
                        except (TypeError, ValueError): # Catch potential formatting errors
                            print(f"{key}: {value}") # Fallback


                # Suggest an optimal length (e.g., 95th percentile)
                # Use .get with a default of np.nan to handle missing keys safely
                p95 = flow_stats.get("95th_percentile", np.nan)
                if not np.isnan(p95):
                   suggested_len = int(p95)
                else:
                   suggested_len = 64 # Default if 95th percentile wasn't calculated
                   logger.warning("95th percentile not found in stats, suggesting default max_seq_len=64.")

                print(f"\nSuggested max_seq_len (e.g., based on 95th percentile): {suggested_len}")
            else:
                print("\nNo flow statistics were calculated.")

        except ImportError as e:
           logger.error(f"Import error: {e}. Check the import statement for 'io_utils' and your Python path/package structure.")
           logger.error("If running as a script, ensure the module structure supports the relative import or adjust the import.")
        except Exception as e:
           logger.exception(f"An unexpected error occurred during execution: {e}")