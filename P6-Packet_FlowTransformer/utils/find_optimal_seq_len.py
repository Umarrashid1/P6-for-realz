import os
import glob
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from . import io_utils


# Configure logging if not already done
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

    # Function to process a single file and extract flow lengths
    def get_lengths_from_file(args: Tuple[pd.DataFrame, str]) -> List[int]:
        df, file_path = args
        flow_lengths = []
        if df is not None and not df.empty:
            if "stream" in df.columns:
                # Group by 'stream' and calculate the size (length) of each group
                try:
                    lengths = df.groupby("stream", sort=False).size()
                    flow_lengths.extend(lengths.tolist())
                except Exception as e:
                    logging.error(f"Error processing flows in {file_path}: {e}")
            else:
                logging.warning(f"'stream' column not found in {file_path}. Cannot calculate flow lengths.")
        return flow_lengths

    # Load files in parallel using ThreadPoolExecutor
    loaded_files_args = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
        # Submit loading tasks
        futures = [
            tpool.submit(io_utils.load_csv_file, fp, test_mode, rows_per_file)
            for fp in all_files
        ]
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                loaded_files_args.append(result) # result is (df, file_path)

    if not loaded_files_args:
        logging.warning("No data loaded from any files.")
        return {}

    logging.info(f"Loaded {len(loaded_files_args)} files. Calculating flow lengths...")

    # Process loaded dataframes in parallel using ProcessPoolExecutor for CPU-bound task
    # (Can also use ThreadPoolExecutor if groupby is not CPU-intensive)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ppool:
        # Submit processing tasks
        futures = [ppool.submit(get_lengths_from_file, args) for args in loaded_files_args]
        # Collect results
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                lengths = future.result()
                all_flow_lengths.extend(lengths)
                if (i + 1) % 10 == 0 or (i + 1) == len(loaded_files_args):
                     logging.info(f"Processed {i+1}/{len(loaded_files_args)} files for lengths...")
            except Exception as e:
                logging.error(f"Error processing future result: {e}")


    if not all_flow_lengths:
        logging.warning("No valid flows found across all processed files.")
        return {}

    # Calculate statistics using numpy
    flow_lengths_np = np.array(all_flow_lengths)
    stats = {
        "count": len(flow_lengths_np),
        "mean": np.mean(flow_lengths_np),
        "std_dev": np.std(flow_lengths_np),
        "min": np.min(flow_lengths_np),
        "max": np.max(flow_lengths_np),
    }

    # Calculate requested percentiles
    for p in percentiles:
        stats[f"{p}th_percentile"] = np.percentile(flow_lengths_np, p)

    logging.info("Flow length analysis complete.")
    logging.info(f"Stats: {stats}")

    return stats

# Example Usage:
if __name__ == "__main__":
    # Configure logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()] # Log to console
    )

    # --- Mock io_utils for demonstration ---
    # Replace this with your actual io_utils import



    # --- End Mock io_utils ---

    DATASET_DIRECTORY = "../../dataset/raw_dataset" # Directory with your CSVs
    flow_stats = calculate_flow_length_stats(DATASET_DIRECTORY, test_mode=False)

    if flow_stats:
        print("\n--- Flow Length Statistics ---")
        for key, value in flow_stats.items():
            print(f"{key}: {value:.2f}")

        # Suggest an optimal length (e.g., 95th percentile)
        suggested_len = int(flow_stats.get("95th_percentile", 64)) # Default to 64 if not found
        print(f"\nSuggested max_seq_len (e.g., based on 95th percentile): {suggested_len}")

        # Clean up dummy files
        import shutil
        # shutil.rmtree(DATASET_DIRECTORY)
        # print(f"Cleaned up {DATASET_DIRECTORY}")

    # Now you can use 'suggested_len' in your main preprocessing function:
    # preprocess_flows_as_sequences(
    #     dataset_dir=DATASET_DIRECTORY,
    #     output_file="processed_data.pt",
    #     max_seq_len=suggested_len
    # )