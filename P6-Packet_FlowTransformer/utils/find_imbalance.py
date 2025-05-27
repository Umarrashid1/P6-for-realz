import os
import glob
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, DefaultDict
from collections import defaultdict
from .io_utils import list_csv_files, load_csv_file



LOG_FILE_NAME = "directory_row_counts.log"

TARGET_DATASET_DIRECTORY = "../../dataset/raw_dataset"


def setup_logging():
    """Configures logging to write to a file."""
    logging.basicConfig(
        filename=LOG_FILE_NAME,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'  # Overwrite log file each time
    )


def count_rows_per_directory(root_dir: str) -> DefaultDict[str, int]:
    """
    Counts the total number of rows in CSV files for each directory
    found within the root_dir and its subdirectories.

    Args:
        root_dir (str): The root directory to scan.

    Returns:
        DefaultDict[str, int]: A dictionary mapping directory paths to their total row counts.
    """
    logging.info(f"Starting row count for directory: {root_dir}")

    # Using defaultdict to simplify summing counts
    directory_row_counts: DefaultDict[str, int] = defaultdict(int)

    # Get all CSV files within the root directory and its subdirectories
    csv_files = list_csv_files(root_dir)

    if not csv_files:
        logging.info(f"No CSV files found in {root_dir} or its subdirectories.")
        return directory_row_counts

    logging.info(f"Found {len(csv_files)} CSV files to process.")

    for file_path in csv_files:

        current_file_directory = os.path.normpath(os.path.dirname(file_path))

        logging.info(f"Processing file: {file_path}")

        # Load the CSV file
        loaded_data = load_csv_file(file_path, test_mode=False)

        if loaded_data:
            df, _ = loaded_data
            num_rows = len(df)
            directory_row_counts[current_file_directory] += num_rows
            logging.info(
                f"Successfully read {num_rows} rows from {file_path}. Directory '{current_file_directory}' total: {directory_row_counts[current_file_directory]}")
        else:
            logging.warning(f"Skipped file (or failed to load): {file_path}")

    return directory_row_counts


def main():
    """
    Main function to set up logging, count rows, and log results.
    """
    setup_logging()
    logging.info("Script started.")


    # Perform the row counting
    all_directory_counts = count_rows_per_directory(TARGET_DATASET_DIRECTORY)

    if not all_directory_counts:
        logging.info("No row counts to report.")
    else:
        logging.info("Directory Row Counts ")
        for dir_path, count in sorted(all_directory_counts.items()):  # Sort for consistent output
            logging.info(f"Directory: {dir_path} - Total Rows: {count}")
        logging.info("End of Report")

    success_message = f"Script finished. Results logged to {LOG_FILE_NAME}"
    print(success_message)
    logging.info(success_message)


if __name__ == "__main__":
    main()