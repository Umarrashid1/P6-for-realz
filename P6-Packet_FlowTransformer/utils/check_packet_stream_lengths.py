import os
import pandas as pd
import numpy as np
import logging
import time
import concurrent.futures
from collections import Counter
from pathlib import Path

# Assuming io_utils.py is in the same directory or accessible in the Python path
# If running this script from the 'utils' directory, and io_utils is also there:
try:
    from . import io_utils
except ImportError:
    # Fallback if running as a standalone script and io_utils is in the same directory
    import io_utils

# --- Configuration ---
# Adjust this path to your raw packet dataset directory
# It's assumed this script is in the 'utils' directory,
# so '..' navigates up to P6-Packet_FlowTransformer, then into the dataset.
DEFAULT_DATASET_DIR = "../../../dataset/raw_dataset"
STREAM_COLUMN_NAME = "stream"  # Column name identifying packet streams
LOG_FILE_NAME = "packet_stream_length_analysis.log"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_NAME, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_packet_stream_lengths(
    dataset_dir: str,
    stream_col_name: str = "stream",
    test_mode: bool = False,
    rows_per_file: int = 0 # 0 means all rows
):
    """
    Analyzes CSV files in a directory to find the distribution of packet stream lengths.
    Streams are grouped across ALL files.

    Args:
        dataset_dir: Path to the directory containing CSV files.
        stream_col_name: The exact name of the column identifying packet streams.
        test_mode: If True, load only a subset of rows specified by rows_per_file (if > 0).
        rows_per_file: Max rows to load per file if test_mode is True and this is > 0.
                       If test_mode is False, this argument is ignored and all rows are loaded
                       unless rows_per_file is set (then it acts as a general limit per file).
    Returns:
        A pandas Series containing the lengths of all unique streams found, or None if error.
    """
    start_time = time.time()
    logger.info(f"Starting packet stream length analysis for directory: {dataset_dir}")
    logger.info(f"Stream column: '{stream_col_name}'")
    if test_mode and rows_per_file > 0:
        logger.info(f"TEST MODE: Processing up to {rows_per_file} rows per file.")
    elif rows_per_file > 0:
        logger.info(f"Processing up to {rows_per_file} rows per file.")
    else:
        logger.info("Processing all rows per file.")

    all_files = io_utils.list_csv_files(dataset_dir)
    if not all_files:
        logger.warning(f"No CSV files found in {dataset_dir}")
        return None

    logger.info(f"Found {len(all_files)} CSV files to process.")

    loaded_stream_series_list: list[pd.Series] = []
    files_processed_count = 0
    files_with_stream_col_count = 0

    # Determine actual nrows to load based on test_mode and rows_per_file
    effective_rows_per_file = None
    if test_mode and rows_per_file > 0:
        effective_rows_per_file = rows_per_file
    elif not test_mode and rows_per_file > 0: # General limit if not test mode
        effective_rows_per_file = rows_per_file


    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        future_to_file = {
            executor.submit(
                io_utils.load_csv_file,
                fp,
                False, # test_mode for load_csv_file controls nrows directly
                effective_rows_per_file if effective_rows_per_file else 0 # Pass 0 if no limit
            ): fp for fp in all_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                load_result = future.result()
                df = None
                if isinstance(load_result, tuple) and len(load_result) > 0 and isinstance(load_result[0], pd.DataFrame):
                    df = load_result[0]
                elif isinstance(load_result, pd.DataFrame): # If load_csv_file was modified to return df directly
                    df = load_result

                files_processed_count += 1
                if df is not None and not df.empty:
                    if stream_col_name in df.columns:
                        # Keep only the stream column to save memory
                        loaded_stream_series_list.append(df[stream_col_name])
                        files_with_stream_col_count +=1
                    else:
                        logger.warning(f"'{stream_col_name}' column not found in {Path(file_path).name}. Skipping this file for analysis.")
                elif df is None:
                     logger.warning(f"Failed to load or got empty result for {Path(file_path).name}. load_csv_file might have skipped it.")


                if files_processed_count % 50 == 0 or files_processed_count == len(all_files):
                    logger.info(f"Loaded {files_processed_count}/{len(all_files)} files...")

            except Exception as e:
                logger.error(f"Error processing file {Path(file_path).name}: {e}", exc_info=False) # Set exc_info=True for full traceback

    if not loaded_stream_series_list:
        logger.error("No data successfully loaded or stream column not found in any file.")
        return None

    logger.info(f"Finished loading data from {files_with_stream_col_count} files that contained the '{stream_col_name}' column.")
    logger.info("Combining data from all files to identify unique streams and their lengths...")

    try:
        # Concatenate all series into one large series
        combined_streams_series = pd.concat(loaded_stream_series_list, ignore_index=True)
        del loaded_stream_series_list # Free memory
        logger.info(f"Data combined. Total packets (rows) across relevant files: {len(combined_streams_series)}.")
    except Exception as e:
        logger.error(f"Error concatenating stream series: {e}")
        return None

    if combined_streams_series.empty:
        logger.warning("Combined stream data is empty. No statistics to calculate.")
        return None

    logger.info(f"Performing global groupby on '{stream_col_name}' to calculate lengths...")
    # Group by the stream identifier across the *entire* dataset and count packets per stream
    # This gives a Series where index is stream_id and value is its length (packet count)
    stream_lengths_series = combined_streams_series.groupby(combined_streams_series, sort=False).size()
    del combined_streams_series # Free memory

    logger.info(f"Found {len(stream_lengths_series)} unique streams across all processed files.")
    total_analysis_time = time.time() - start_time
    logger.info(f"Stream length calculation complete. Total time: {total_analysis_time:.2f} seconds.")

    return stream_lengths_series

def print_statistics(stream_lengths: pd.Series):
    if stream_lengths is None or stream_lengths.empty:
        logger.info("No stream lengths data to generate statistics for.")
        return

    logger.info("\n--- Packet Stream Length Statistics ---")
    logger.info(f"Total Unique Streams: {len(stream_lengths)}")
    logger.info(f"Mean Length: {stream_lengths.mean():.2f} packets")
    logger.info(f"Median Length: {stream_lengths.median():.2f} packets")
    logger.info(f"Standard Deviation: {stream_lengths.std():.2f} packets")
    logger.info(f"Min Length: {stream_lengths.min()} packets")
    logger.info(f"Max Length: {stream_lengths.max()} packets")

    logger.info("\nPercentiles for Stream Lengths:")
    percentiles_to_calc = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    for p in percentiles_to_calc:
        logger.info(f"  {p}th percentile: {stream_lengths.quantile(p/100):.2f} packets")

    logger.info("\nCounts for Specific Short Stream Lengths:")
    length_counts = Counter(stream_lengths)
    max_len_to_show_counts = 10 # Show counts for lengths 1 through 10
    for i in range(1, max_len_to_show_counts + 1):
        count = length_counts.get(i, 0)
        percentage = (count / len(stream_lengths)) * 100 if len(stream_lengths) > 0 else 0
        logger.info(f"  Streams of length {i}: {count} ({percentage:.2f}%)")

    if length_counts.get(1,0) > 0 :
        logger.info(f"\nNOTE: Your observation about many streams having length 1 is confirmed. Count: {length_counts.get(1,0)}")


if __name__ == "__main__":
    dataset_path = DEFAULT_DATASET_DIR
    # You can override the dataset_path from command line if needed, e.g.
    # import sys
    # if len(sys.argv) > 1:
    #   dataset_path = sys.argv[1]

    if not Path(dataset_path).is_dir():
        logger.error(f"Dataset directory not found: {Path(dataset_path).resolve()}")
        logger.error("Please ensure DEFAULT_DATASET_DIR is correct or provide a valid path.")
    else:
        logger.info(f"Using dataset directory: {Path(dataset_path).resolve()}")
        stream_lengths_data = analyze_packet_stream_lengths(
            dataset_dir=dataset_path,
            stream_col_name=STREAM_COLUMN_NAME,
            test_mode=False, # Set to True for a quick test on a subset of rows
            rows_per_file=0 # Set to e.g. 10000 if test_mode=True for faster testing
        )
        print_statistics(stream_lengths_data)
        logger.info(f"Analysis complete. Log saved to {LOG_FILE_NAME}")