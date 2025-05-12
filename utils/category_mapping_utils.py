# utils/category_mapping_utils.py
import os
import json
import pandas as pd
import concurrent.futures
from pathlib import Path
import ast  # For literal_eval
from typing import List, Dict, Any
import logging

# Assuming io_utils is in the same directory or accessible in PYTHONPATH
from . import io_utils


# Configure basic logging for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (category_mapping_utils) %(message)s')

def generate_and_save_category_mappings(
        dataset_dir: str,
        categorical_columns: List[str],  # Explicitly pass the list of columns
        output_mapping_path: str = "category_mappings.json",
        test_mode: bool = False,
        rows_per_file: int = 20_000,  # Default from your script
        num_workers: Optional[int] = None
) -> Dict[str, Dict[Any, int]]:
    """
    Generates category-to-integer mappings for specified columns from CSV files
    in a directory and saves them to a JSON file.

    Keys in the saved JSON for each category value are their string representation
    obtained by repr(), e.g., repr('some_string'), repr(123).

    Args:
        dataset_dir: Directory containing the CSV files.
        categorical_columns: List of column names to generate mappings for.
        output_mapping_path: Path to save the generated JSON mappings.
        test_mode: If True, limits rows read per file.
        rows_per_file: Max rows to read per file if test_mode is True.
        num_workers: Number of workers for concurrent file loading. Defaults to os.cpu_count().

    Returns:
        The generated mappings dictionary.
    """
    logging.info(f"Starting category mapping generation for directory: {dataset_dir}")
    logging.info(f"Target categorical columns: {categorical_columns}")
    logging.info(f"Output will be saved to: {output_mapping_path}")

    if not categorical_columns:
        logging.warning("No categorical columns provided. Returning empty mappings.")
        return {}

    all_files = io_utils.list_csv_files(dataset_dir, recursive=True)
    if not all_files:
        logging.error(f"No CSV files found in {dataset_dir}. Cannot generate mappings.")
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    if num_workers is None:
        num_workers = os.cpu_count()
    logging.info(f"Using {num_workers} workers for loading CSV files.")

    # ── 1. Load relevant columns from CSVs concurrently ─────────────────────
    loaded_data_for_columns: Dict[str, List[pd.Series]] = {col: [] for col in categorical_columns}
    files_processed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Submit tasks to load only the required categorical columns
        # io_utils.load_csv_to_df can take required_cols argument
        futures = [
            pool.submit(io_utils.load_csv_to_df, fp, categorical_columns, test_mode, rows_per_file)
            for fp in all_files
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                df, file_path = result  # df here contains only categorical_columns
                for col in categorical_columns:
                    if col in df:
                        loaded_data_for_columns[col].append(df[col])
                files_processed_count += 1
                # logging.debug(f"Loaded categorical columns from: {Path(file_path).name}")

    logging.info(f"Successfully loaded data from {files_processed_count} files.")

    if not any(loaded_data_for_columns.values()):  # Check if any data was loaded for any column
        logging.error("No data loaded for any specified categorical columns. Cannot build mappings.")
        raise RuntimeError("No data loaded for categorical columns to build mappings.")

    # ── 2. Build mappings column‑wise ──────────────────────────────────────
    final_mappings: Dict[str, Dict[Any, int]] = {}  # Stores original_value -> code
    json_savable_mappings: Dict[str, Dict[str, int]] = {}  # Stores repr(original_value) -> code

    for col in categorical_columns:
        if not loaded_data_for_columns[col]:
            logging.warning(f"No data found for column '{col}'. Skipping mapping generation for it.")
            final_mappings[col] = {repr("unknown"): 0}  # Default mapping with only unknown
            json_savable_mappings[col] = {repr("unknown"): 0}
            continue

        # Concatenate all series for the current column and find unique values
        combined_series = pd.concat(loaded_data_for_columns[col], ignore_index=True)
        # Convert to string before finding unique values to handle mixed types, then drop NaNs
        # NaNs will be handled by the fillna("unknown") step in the main preprocessing later
        # For mapping generation, we want to map actual present values.
        unique_values = combined_series.astype(str).dropna().unique().tolist()

        # Ensure "unknown" is a defined category, typically last.
        # Remove any existing "unknown" (case-insensitive) to control its position.
        processed_unique_values = []
        unknown_present_in_data = False
        for v in unique_values:
            if str(v).lower() == "unknown":
                unknown_present_in_data = True
            else:
                processed_unique_values.append(v)

        # Sort values for consistent mapping generation (important for reproducibility)
        # Try to sort, handling mixed types by converting to string for sorting key
        try:
            processed_unique_values.sort(key=lambda x: str(x))
        except TypeError as e:
            logging.warning(
                f"Could not sort unique values for column '{col}' due to mixed types: {e}. Order may be inconsistent.")
            # Fallback to unsorted if mixed type sorting fails robustly

        # Add "unknown" as the last category if it wasn't the only value or to ensure it exists
        # The main preprocessing script will fill NaNs with "unknown" string before applying these mappings.
        if "unknown" not in [str(v).lower() for v in processed_unique_values]:
            processed_unique_values.append("unknown")  # Add the string "unknown"
        elif not unknown_present_in_data and "unknown" in [str(v).lower() for v in processed_unique_values]:
            # This case means "unknown" was added by sorting but wasn't explicitly in data.
            # It's fine, it will be the last one.
            pass

        # Create mapping: original_value -> integer_code
        # Create JSON-savable mapping: repr(original_value) -> integer_code
        # The `load_mappings` function will use ast.literal_eval to convert repr(original_value) back.
        current_col_mapping: Dict[Any, int] = {}
        current_col_json_mapping: Dict[str, int] = {}

        for idx, val in enumerate(processed_unique_values):
            # ast.literal_eval might fail if val is a complex object not representable by a literal.
            # For typical CSV data (strings, numbers), this should be fine.
            # If val is already a string "unknown", repr("unknown") is "'unknown'".
            # If val is an int 123, repr(123) is "123".
            try:
                # The key for the JSON mapping is the string representation of the value
                # This is what your original script did and is robust for JSON.
                key_for_json = repr(val)
                current_col_mapping[val] = idx
                current_col_json_mapping[key_for_json] = idx
            except Exception as e:
                logging.error(
                    f"Could not generate repr for value '{val}' in column '{col}'. Skipping this value. Error: {e}")
                continue

        final_mappings[col] = current_col_mapping
        json_savable_mappings[col] = current_col_json_mapping

        # Sanity check: ensure 'unknown' (as a string) is mapped and is last if added
        # The key in final_mappings would be the actual string "unknown"
        # The key in json_savable_mappings would be repr("unknown")
        unknown_string_key = "unknown"
        unknown_repr_key = repr("unknown")

        if unknown_string_key in current_col_mapping:
            if current_col_mapping[unknown_string_key] != len(processed_unique_values) - 1:
                logging.warning(
                    f"For column '{col}', 'unknown' string was mapped but not as the last index. This might be unexpected if 'unknown' was intended as a catch-all.")
        elif unknown_repr_key in current_col_json_mapping:
            if current_col_json_mapping[unknown_repr_key] != len(processed_unique_values) - 1:
                logging.warning(f"For column '{col}', repr('unknown') was mapped but not as the last index.")
        else:
            logging.warning(
                f"For column '{col}', an 'unknown' category was not explicitly mapped. Ensure your data or fillna strategy aligns.")

        logging.info(f"[MAPPING] Column '{col}': {len(current_col_mapping)} categories generated.")

    # ── 3. Save JSON-savable mappings to disk ────────────────────────────────
    try:
        output_path_obj = Path(output_mapping_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        with open(output_path_obj, "w") as f:
            json.dump(json_savable_mappings, f, indent=2)
        logging.info(f"Successfully saved category mappings to {output_mapping_path}")
    except Exception as e:
        logging.error(f"Failed to save mappings to {output_mapping_path}: {e}")
        raise

    return final_mappings  # Return the Python-usable mappings


def load_mappings(path: str = "category_mappings.json") -> Dict[str, Dict[Any, int]]:
    """
    Loads category mappings from a JSON file.
    Keys in the loaded mapping dictionary for each category value are converted
    from their string representation (repr(value)) back to their original Python types
    using ast.literal_eval().

    Args:
        path: Path to the JSON mapping file.

    Returns:
        A dictionary where keys are column names, and values are dictionaries
        mapping original category values (e.g., strings, numbers) to integer codes.
    """
    logging.info(f"Loading category mappings from: {path}")
    try:
        with open(path, "r") as f:
            # raw_mappings_from_json has keys like "src_ip" and values like {"'192.168.1.1'": 0, "'10.0.0.1'": 1}
            raw_mappings_from_json = json.load(f)
    except FileNotFoundError:
        logging.error(f"Mappings file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path}: {e}")
        raise

    # This will be the final Python-usable mapping: {col_name: {original_value: code}}
    python_usable_mappings: Dict[str, Dict[Any, int]] = {}
    for col_name, mapping_repr_keys in raw_mappings_from_json.items():
        current_col_python_map: Dict[Any, int] = {}
        for key_as_repr_string, value_as_int_code in mapping_repr_keys.items():
            try:
                # Convert the string representation of the key back to its original Python object
                # e.g., "'192.168.1.1'" becomes "192.168.1.1" (string)
                # e.g., "'123'" becomes "123" (string), or "123" (if it was repr(123)) becomes 123 (int)
                # e.g., "None" becomes None
                original_key_value = ast.literal_eval(key_as_repr_string)
                current_col_python_map[original_key_value] = value_as_int_code
            except (ValueError, SyntaxError, TypeError) as e:
                # Log if a key can't be evaluated, but continue
                logging.warning(
                    f"Could not evaluate key '{key_as_repr_string}' for column '{col_name}' in mapping file '{path}'. Skipping this key. Error: {e}")
                continue  # Skip problematic keys
        python_usable_mappings[col_name] = current_col_python_map

    logging.info(f"Successfully loaded and parsed {len(python_usable_mappings)} column mappings.")
    return python_usable_mappings


if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (cat_map_test) %(message)s')

    # --- Configuration for testing generate_and_save_category_mappings ---
    # Create dummy data and config for testing
    sample_data_dir = Path("sample_cat_map_data")
    sample_data_dir.mkdir(exist_ok=True)
    (sample_data_dir / "sub").mkdir(exist_ok=True)

    # Define which columns are categorical for this test
    test_categorical_columns = ['protocol', 'http_method', 'port_type', 'mixed_col']

    # Create dummy CSV files
    df1_data = {
        'protocol': ['TCP', 'UDP', 'TCP', 'ICMP', None],
        'http_method': ['GET', 'POST', 'GET', np.nan, 'PUT'],
        'port_type': ['http', 'dns', 'http', 'other', 'unknown'],
        'mixed_col': [10, 'text', 10, 20.5, 'text'],
        'numerical_col': [1.0, 2.5, 3.0, 4.5, 5.5]
    }
    pd.DataFrame(df1_data).to_csv(sample_data_dir / "data1.csv", index=False)

    df2_data = {
        'protocol': ['UDP', 'UDP', 'TCP', 'ARP'],
        'http_method': ['GET', None, 'DELETE', 'GET'],
        'port_type': ['dns', 'dns', 'https', 'other'],
        'mixed_col': ['new_val', 30, 10, None],  # New value and an existing one
        'numerical_col': [6.0, 7.5, 8.0, 9.5]
    }
    pd.DataFrame(df2_data).to_csv(sample_data_dir / "sub" / "data2.csv", index=False)

    # Path for the generated mappings
    test_mapping_file = Path("test_category_mappings.json")

    print("\n--- Testing generate_and_save_category_mappings ---")
    try:
        generated_mappings = generate_and_save_category_mappings(
            dataset_dir=str(sample_data_dir),
            categorical_columns=test_categorical_columns,
            output_mapping_path=str(test_mapping_file),
            test_mode=False  # Process all rows for mapping generation
        )
        print(f"\nGenerated mappings (Python dict returned by function):")
        for col, mapping in generated_mappings.items():
            print(f"  Column '{col}': {mapping}")

        print(f"\nCheck the content of '{test_mapping_file}' to see the JSON output.")

        print("\n--- Testing load_mappings ---")
        loaded_mappings_from_file = load_mappings(path=str(test_mapping_file))
        print(f"\nLoaded mappings (from file, keys parsed by ast.literal_eval):")
        for col, mapping in loaded_mappings_from_file.items():
            print(f"  Column '{col}':")
            for k, v in mapping.items():
                print(f"    {k} (type: {type(k)}): {v}")

        # Verification
        assert len(generated_mappings) == len(loaded_mappings_from_file)
        for col in generated_mappings:
            assert col in loaded_mappings_from_file
            # Compare item by item as dict order might not be guaranteed for very old Pythons
            # For Python 3.7+ dict order is insertion order, so direct comparison should be fine
            # if generated_mappings[col] != loaded_mappings_from_file[col]:
            #     print(f"Mismatch in column {col}")
            #     print("Generated:", generated_mappings[col])
            #     print("Loaded:   ", loaded_mappings_from_file[col])
            # assert generated_mappings[col] == loaded_mappings_from_file[col]
            # More robust check:
            assert len(generated_mappings[col]) == len(loaded_mappings_from_file[col])
            for k_gen, v_gen in generated_mappings[col].items():
                assert k_gen in loaded_mappings_from_file[col]
                assert loaded_mappings_from_file[col][k_gen] == v_gen

        print("\n✅ All tests seemed to pass based on structure and content.")

    except Exception as e:
        logging.error(f"Error during test: {e}", exc_info=True)
    finally:
        # Clean up dummy data and mapping file
        import shutil

        # shutil.rmtree(sample_data_dir, ignore_errors=True)
        # if test_mapping_file.exists():
        #     test_mapping_file.unlink()
        print(f"\n(To clean up, manually delete: {sample_data_dir} and {test_mapping_file})")

