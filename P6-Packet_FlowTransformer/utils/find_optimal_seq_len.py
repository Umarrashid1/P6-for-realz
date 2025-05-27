import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import List, Tuple, Dict, Optional
import logging
import time
from . import io_utils

def calculate_global_flow_length_stats(
    dataset_dir: str,
    stream_col_name: str = "stream",
    test_mode: bool = False,
    rows_per_file: Optional[int] = None,
    percentiles: List[int] = [50, 75, 90, 95, 99, 100]
) -> Dict[str, float]:
    """
    Analyzes CSV files in a directory to find the distribution of flow lengths,
    grouping flows across ALL files.

    Args:
        dataset_dir: Path to the directory containing CSV files.
        stream_col_name: The exact name of the column identifying flows/streams.
        test_mode: If True, load only a small subset of rows for quick testing.
        rows_per_file: Maximum rows to load per file (None for all rows).
                       Useful for large files if test_mode is False.
        percentiles: A list of percentiles to calculate for flow lengths.

    Returns:
        A dictionary containing statistics (count, mean, std, min, max,
        and specified percentiles) of the flow lengths across the entire dataset.
        Returns an empty dictionary if no valid flows are found.
    """
    start_time = time.time()
    all_files = io_utils.list_csv_files(dataset_dir)
    if not all_files:
        logging.warning(f"No CSV files found in {dataset_dir}")
        return {}

    logging.info(f"Found {len(all_files)} files in {dataset_dir}. Starting data loading...")

    loaded_data_frames: List[pd.DataFrame] = [] # Store loaded DataFrames (or just Series)

    # --- Parallel Loading Phase ---
    # Use slightly fewer workers than CPUs for loading if IO is the bottleneck,
    # or leave as os.cpu_count() if unsure.
    # Consider adjusting based on HPC node specifics (cores vs threads)
    num_load_workers = os.cpu_count() or 8 # Default to 8 if os.cpu_count() fails
    logging.info(f"Using {num_load_workers} workers for loading files.")

    files_processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as tpool:
        # Submit loading tasks
        # *** Optimization Note ***:
        # If io_utils.load_csv_file supports a 'usecols' argument, use it:
        # futures_load = {
        #     tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file, usecols=[stream_col_name]): fp
        #     for fp in all_files
        # }
        # If not, we load potentially more data than needed initially.
        futures_load = {
            tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file): fp
            for fp in all_files
        }

        # Collect results as they complete
        total_files = len(all_files)
        for future in concurrent.futures.as_completed(futures_load):
            file_path = futures_load[future] # Get path for context
            try:
                load_result = future.result()

                # Assuming result is the DataFrame or (DataFrame, path_string)
                df = None
                if isinstance(load_result, tuple) and len(load_result) > 0 and isinstance(load_result[0], pd.DataFrame):
                    df = load_result[0]
                elif isinstance(load_result, pd.DataFrame):
                    df = load_result

                if df is not None:
                    if not df.empty:
                        if stream_col_name in df.columns:
                            # Keep only the essential column to save memory before combining
                            loaded_data_frames.append(df[[stream_col_name]])
                        else:
                             logging.warning(f"'{stream_col_name}' column not found in {file_path}. Skipping file for analysis.")

                else:
                    logging.warning(f"Failed to load or got empty result for {file_path}")

                files_processed_count += 1
                if files_processed_count % 20 == 0 or files_processed_count == total_files:
                    logging.info(f"Loaded {files_processed_count}/{total_files} files...")

            except Exception as e:
                logging.error(f"Error loading or processing file {file_path}: {e}")

    if not loaded_data_frames:
        logging.error("No data successfully loaded or stream column not found in any file.")
        return {}

    loading_done_time = time.time()
    logging.info(f"Finished loading data ({len(loaded_data_frames)} files with '{stream_col_name}' column). Time elapsed: {loading_done_time - start_time:.2f} seconds.")
    logging.info("Combining data from all files...")

    # --- Combine Data Phase ---
    try:
        combined_df = pd.concat(loaded_data_frames, ignore_index=True)
        # Free up memory by deleting the list of individual frames
        del loaded_data_frames
        logging.info(f"Data combined. Total packets (rows): {len(combined_df)}. Calculating global flow lengths...")
    except Exception as e:
        logging.error(f"Error concatenating DataFrames: {e}")
        return {}

    combine_done_time = time.time()
    logging.info(f"Data combination took: {combine_done_time - loading_done_time:.2f} seconds.")

    # --- Global GroupBy and Statistics Phase ---
    if combined_df.empty:
        logging.warning("Combined DataFrame is empty. No statistics to calculate.")
        return {}

    try:
        logging.info(f"Performing global groupby on '{stream_col_name}'...")
        # Group by the stream identifier across the *entire* dataset and count packets per stream
        flow_lengths_series = combined_df.groupby(stream_col_name, sort=False).size()
        del combined_df # Free memory again

        groupby_done_time = time.time()
        logging.info(f"Global groupby complete. Found {len(flow_lengths_series)} unique flows. Time elapsed: {groupby_done_time - combine_done_time:.2f} seconds.")

        if flow_lengths_series.empty:
             logging.warning("No flows found after grouping. No statistics calculated.")
             return {}

        # Calculate statistics using numpy for potentially better performance on large series
        logging.info("Calculating final statistics...")
        flow_lengths_np = flow_lengths_series.to_numpy(dtype=np.int64) # Convert to numpy array
        del flow_lengths_series # Free memory

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
            except IndexError: # Should not happen if flow_lengths_np is not empty
                logging.warning(f"Could not calculate {p}th percentile.")
                stats[f"{p}th_percentile"] = np.nan

        stats_done_time = time.time()
        logging.info(f"Statistics calculation took: {stats_done_time - groupby_done_time:.2f} seconds.")
        logging.info(f"Total analysis time: {stats_done_time - start_time:.2f} seconds.")
        logging.info("Global flow length analysis complete.")

        # Format stats for logging
        formatted_stats = {
            k: f'{v:.2f}' if isinstance(v, (float, np.number)) and not np.isnan(v) else
               ('NaN' if isinstance(v, float) and np.isnan(v) else v)
            for k, v in stats.items()
        }
        logging.info(f"Stats calculated: {formatted_stats}")

        return stats

    except Exception as e:
        logging.error(f"An error occurred during global grouping or statistics calculation: {e}")
        # If combined_df still exists, maybe log its memory usage?
        # import sys
        # logging.error(f"Combined DF memory usage: {sys.getsizeof(combined_df) / (1024**3):.2f} GB")
        return {}


# --- Example Usage (Updated) ---
if __name__ == "__main__":
    # Configure logging for the example script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()] # Log to console
    )
    logger = logging.getLogger(__name__)

    # --- IMPORTANT ---
    # Adjust this path relative to where you RUN the script from,
    # or use an absolute path.
    DATASET_DIRECTORY = "../../dataset/raw_dataset" # Adjust as needed!
    STREAM_COLUMN = "stream" # ** Specify the correct column name here **

    # Check if directory exists
    if not os.path.isdir(DATASET_DIRECTORY):
       logger.error(f"Dataset directory not found: {os.path.abspath(DATASET_DIRECTORY)}")
       logger.error("Please ensure the DATASET_DIRECTORY path is correct relative to the script execution location.")
    else:
        logger.info(f"Attempting to analyze dataset globally at: {os.path.abspath(DATASET_DIRECTORY)}")
        try:
            # *** Mock io_utils if needed for structural testing ***
            # class MockIoUtils:
            #     def list_csv_files(self, path):
            #         # Create dummy files for testing structure
            #         p = Path(path)
            #         p.mkdir(exist_ok=True)
            #         dummy_files = []
            #         for i in range(5): # Create 5 dummy files
            #             fn = p / f"test_{i}.csv"
            #             # Create very small CSVs with the stream column
            #             pd.DataFrame({
            #                 STREAM_COLUMN: [f'flow_{i}', f'flow_{i}', f'flow_{(i+1)%3}'],
            #                 'other_col': [1,2,3]
            #             }).to_csv(fn, index=False)
            #             dummy_files.append(str(fn))
            #         return dummy_files
            #
            #     def load_csv_file(self, fp, test_mode, rows_per_file, **kwargs):
            #         # Mock loading, respecting usecols if passed (though not explicitly here)
            #         # print(f"Mock loading: {fp}") # Debug print
            #         try:
            #             # Read only necessary cols if specified, otherwise all
            #             usecols = kwargs.get('usecols', None)
            #             nrows = rows_per_file if test_mode or rows_per_file else None
            #             df = pd.read_csv(fp, usecols=usecols, nrows=nrows)
            #             return df # Return only df, not tuple, matching adjusted code
            #         except Exception as e:
            #             print(f"Mock load error for {fp}: {e}")
            #             return None # Simulate loading failure

            # Comment out the real import and uncomment below to use mock
            # import io_utils # Make sure the real one is not active
            # io_utils = MockIoUtils()
            # DATASET_DIRECTORY = "./temp_mock_data" # Use a temp dir for mock data

            # *** Call the updated function ***
            flow_stats = calculate_global_flow_length_stats(
                DATASET_DIRECTORY,
                stream_col_name=STREAM_COLUMN,
                test_mode=False # Set to True for quick functional test
                # rows_per_file=1000 # Uncomment to limit rows per file for testing
            )

        if flow_stats:
            print("\nGlobal Flow Length Statistics")
            for key, value in flow_stats.items():
                if isinstance(value, float) and np.isnan(value):
                    print(f"{key}: NaN")
                else:
                    try:
                        if isinstance(value, (int, float, np.number)):
                            print(f"{key}: {value:,.2f}")
                        else:
                            print(f"{key}: {value}")
                    except (TypeError, ValueError):
                        print(f"{key}: {value}")

                # Suggest an optimal length (e.g., 95th or 99th percentile)
                p_suggest = 99 # Consider using 99th percentile for sequence length
                p_key = f"{p_suggest}th_percentile"
                p_val = flow_stats.get(p_key, np.nan)

            if not np.isnan(p_val):
               suggested_len = int(p_val)
               print(f"\nSuggested max_seq_len (based on {p_key}): {suggested_len}")
            else:
               # Fallback if percentile calculation failed
               p95_val = flow_stats.get("95th_percentile", np.nan)
               if not np.isnan(p95_val):
                   suggested_len = int(p95_val)
                   logger.warning(f"{p_key} not found, using 95th percentile.")
                   print(f"\nSuggested max_seq_len (based on 95th_percentile): {suggested_len}")
               else:
                   suggested_len = 128
                   logger.warning(f"{p_key} and 95th percentile not found, suggesting default max_seq_len={suggested_len}.")
                   print(f"\nSuggested max_seq_len (default): {suggested_len}")

