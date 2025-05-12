# utils/io_utils.py
import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import logging


# Configure basic logging for this module if it's run standalone or imported early
# More sophisticated logging might be handled by the main application
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (io_utils) %(message)s')

def list_csv_files(dataset_dir: str, recursive: bool = True) -> List[str]:
    """
    Lists all CSV files in the given directory.

    Args:
        dataset_dir: The root directory to search for CSV files.
        recursive: If True, searches subdirectories as well.

    Returns:
        A list of absolute paths to the found CSV files.
    """
    csv_files = []
    if not os.path.isdir(dataset_dir):
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return csv_files

    if recursive:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(dataset_dir):
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(dataset_dir, file))

    logging.info(f"Found {len(csv_files)} CSV files in '{dataset_dir}' (recursive={recursive}).")
    return sorted(csv_files)  # Sort for consistent processing order


def load_csv_to_df(
        file_path: str,
        required_cols: Optional[List[str]] = None,
        test_mode: bool = False,
        rows_per_file: Optional[int] = None,
        low_memory: bool = False
) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Loads a single CSV file into a Pandas DataFrame.

    Args:
        file_path: Absolute path to the CSV file.
        required_cols: An optional list of columns that must be present.
                       If None, all columns are loaded. If specified, only these are loaded.
        test_mode: If True and rows_per_file is set, limits the number of rows read.
        rows_per_file: Maximum number of rows to read if test_mode is True.
                       If None or test_mode is False, all rows are read.
        low_memory: Passed to pd.read_csv. Set to False for potentially better type inference
                    on mixed-type columns, but uses more memory.

    Returns:
        A tuple (DataFrame, file_path) if successful, otherwise None.
    """
    try:
        # Determine nrows for pd.read_csv
        nrows_to_load = rows_per_file if test_mode and rows_per_file is not None else None

        # Check header first if required_cols are specified
        if required_cols:
            df_header = pd.read_csv(file_path, nrows=0)  # Read only header
            cols_present_in_header = df_header.columns.tolist()
            cols_missing_in_header = [col for col in required_cols if col not in cols_present_in_header]

            if cols_missing_in_header:
                logging.warning(
                    f"[SKIP LOAD] Missing required columns in {Path(file_path).name}: {cols_missing_in_header}")
                return None

            # If all required columns are present, load only them
            df = pd.read_csv(
                file_path,
                usecols=required_cols,
                nrows=nrows_to_load,
                low_memory=low_memory
            )
        else:
            # Load all columns if required_cols is None
            df = pd.read_csv(
                file_path,
                nrows=nrows_to_load,
                low_memory=low_memory
            )

        if df.empty:
            logging.warning(
                f"[SKIP LOAD] Loaded empty DataFrame from {Path(file_path).name} (nrows_to_load={nrows_to_load}).")
            return None

        # logging.debug(f"Successfully loaded {len(df)} rows from {Path(file_path).name}")
        return df, file_path

    except pd.errors.EmptyDataError:
        logging.warning(f"[SKIP LOAD] Empty CSV file: {Path(file_path).name}")
        return None
    except FileNotFoundError:
        logging.error(f"[ERROR LOAD] File not found: {file_path}")
        return None
    except ValueError as ve:  # Catches errors like "usecols do not match columns"
        logging.error(f"[ERROR LOAD] Value error loading {Path(file_path).name}: {ve}")
        return None
    except Exception as e:
        logging.error(f"[ERROR LOAD] Could not load CSV {Path(file_path).name}: {type(e).__name__} - {e}")
        return None


if __name__ == '__main__':
    # Example Usage (assumes a 'sample_data' directory with some CSVs)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (io_utils_test) %(message)s')

    # Create dummy data for testing
    sample_dir = Path("sample_data_io_test")
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "sub").mkdir(exist_ok=True)

    pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(sample_dir / "file1.csv", index=False)
    pd.DataFrame({'C': [5, 6], 'D': [7, 8]}).to_csv(sample_dir / "file2.csv", index=False)
    pd.DataFrame({'A': [9, 10], 'E': [11, 12]}).to_csv(sample_dir / "sub" / "file3.csv", index=False)
    (sample_dir / "empty.csv").touch()

    print("\n--- Testing list_csv_files (recursive) ---")
    files_recursive = list_csv_files(str(sample_dir), recursive=True)
    for f in files_recursive:
        print(f)

    print("\n--- Testing list_csv_files (non-recursive) ---")
    files_non_recursive = list_csv_files(str(sample_dir), recursive=False)
    for f in files_non_recursive:
        print(f)

    print("\n--- Testing load_csv_to_df ---")
    if files_recursive:
        # Test loading all columns
        result = load_csv_to_df(files_recursive[0])
        if result:
            df, fp = result
            print(f"\nLoaded all columns from {Path(fp).name}:\n{df.head()}")

        # Test loading specific columns (successful)
        result_cols_a_b = load_csv_to_df(files_recursive[0], required_cols=['A', 'B'])
        if result_cols_a_b:
            df, fp = result_cols_a_b
            print(f"\nLoaded specific columns ['A', 'B'] from {Path(fp).name}:\n{df.head()}")

        # Test loading specific columns (one missing)
        result_cols_a_x = load_csv_to_df(files_recursive[0], required_cols=['A', 'X'])  # X is missing
        if result_cols_a_x is None:
            print(f"\nCorrectly skipped loading {Path(files_recursive[0]).name} due to missing column 'X'.")

        # Test test_mode and rows_per_file
        result_test_mode = load_csv_to_df(files_recursive[0], test_mode=True, rows_per_file=1)
        if result_test_mode:
            df, fp = result_test_mode
            print(f"\nLoaded in test mode (1 row) from {Path(fp).name}:\n{df.head()}")

        # Test loading empty file
        result_empty = load_csv_to_df(str(sample_dir / "empty.csv"))
        if result_empty is None:
            print(f"\nCorrectly handled empty file 'empty.csv'.")

    # Clean up dummy data
    import shutil

    # shutil.rmtree(sample_dir)
    print(f"\n(To clean up, manually delete: {sample_dir})")

