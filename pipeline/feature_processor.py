# pipeline/feature_processor.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging


# Configure basic logging for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (feature_processor) %(message)s')

def load_standardization_stats(stats_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads standardization statistics (mean, std, median, clip values, column list)
    from an .npz file.

    Args:
        stats_path: Path to the .npz file containing the statistics.
                    Expected keys in .npz: 'cols', 'mean', 'std', 'median', 'clip_low', 'clip_high'.

    Returns:
        A dictionary containing the loaded statistics if successful, otherwise None.
        The dictionary structure:
        {
            "cols": List[str],
            "mean": Dict[str, float],
            "std": Dict[str, float],
            "median": Dict[str, float],
            "clip_low": Dict[str, float],
            "clip_high": Dict[str, float]
        }
    """
    logging.info(f"Loading standardization stats from: {stats_path}")
    try:
        npz_data = np.load(stats_path)

        required_keys = ['cols', 'mean', 'std', 'median', 'clip_low', 'clip_high']
        for key in required_keys:
            if key not in npz_data:
                logging.error(f"Missing key '{key}' in standardization stats file: {stats_path}")
                return None

        std_cols = npz_data["cols"].tolist()

        stats = {
            "cols": std_cols,
            "mean": dict(zip(std_cols, npz_data["mean"])),
            "std": dict(zip(std_cols, npz_data["std"])),
            "median": dict(zip(std_cols, npz_data["median"])),
            "clip_low": dict(zip(std_cols, npz_data["clip_low"])),
            "clip_high": dict(zip(std_cols, npz_data["clip_high"]))
        }
        logging.info(f"Successfully loaded standardization stats for {len(std_cols)} columns.")
        return stats
    except FileNotFoundError:
        logging.error(f"Standardization stats file not found: {stats_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading standardization stats from {stats_path}: {e}")
        return None


def process_numerical_features(
        df: pd.DataFrame,
        numerical_cols: List[str],
        stats: Dict[str, Any],  # Loaded from load_standardization_stats
        eps: float = 1e-6
) -> pd.DataFrame:
    """
    Applies clipping, NaN filling (with median), and standardization to numerical columns.

    Args:
        df: Input Pandas DataFrame.
        numerical_cols: List of numerical column names to process.
        stats: Dictionary of standardization statistics (mean, std, median, clip values).
               Must contain 'clip_low', 'clip_high', 'median', 'mean', 'std' as sub-dictionaries
               keyed by column name.
        eps: Epsilon value to add to standard deviation to prevent division by zero.

    Returns:
        DataFrame with processed numerical columns.
    """
    logging.info(f"Processing {len(numerical_cols)} numerical features...")
    df_processed = df.copy()

    # Extract stats dictionaries for easier access
    clip_low_map = stats.get("clip_low", {})
    clip_high_map = stats.get("clip_high", {})
    median_map = stats.get("median", {})
    mean_map = stats.get("mean", {})
    std_map = stats.get("std", {})
    stats_cols_available = stats.get("cols", [])

    for col in numerical_cols:
        if col not in df_processed.columns:
            logging.warning(f"Numerical column '{col}' not found in DataFrame. Skipping.")
            continue

        # Ensure column is numeric, coercing errors
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Check if stats are available for this column
        if col not in stats_cols_available:
            logging.warning(f"No standardization stats available for numerical column '{col}'. "
                            "Skipping clipping, median fill, and standardization for this column. "
                            "Only 'coerce to numeric' applied.")
            continue  # Skip full processing if no stats

        # 1. Clip
        if col in clip_low_map and col in clip_high_map:
            df_processed[col] = df_processed[col].clip(lower=clip_low_map[col], upper=clip_high_map[col])
        else:
            logging.warning(f"Clipping stats not found for '{col}'. Skipping clipping.")

        # 2. Fill NaN (using median from stats)
        if col in median_map:
            df_processed[col].fillna(median_map[col], inplace=True)
        else:
            logging.warning(
                f"Median stat not found for '{col}'. NaNs in this column might persist or be handled by global fillna if any.")
            # If NaNs still exist after this, downstream processes might fail.
            # Consider a default fill if median is missing, e.g., df_processed[col].fillna(0, inplace=True)

        # 3. Standardize (using mean and std from stats)
        if col in mean_map and col in std_map:
            mean_val = mean_map[col]
            std_val = std_map[col]
            df_processed[col] = (df_processed[col] - mean_val) / (std_val + eps)
        else:
            logging.warning(f"Mean or std stat not found for '{col}'. Skipping standardization.")

        # Final check for NaNs/Infs in the processed column
        if df_processed[col].isnull().any() or np.isinf(df_processed[col]).any():
            logging.warning(
                f"NaNs or Infs still present in numerical column '{col}' after processing. Consider default fill or check stats.")
            # Example: df_processed[col].fillna(0, inplace=True) # Default fill for remaining NaNs

    return df_processed


def encode_categorical_features(
        df: pd.DataFrame,
        categorical_cols: List[str],
        category_mappings: Dict[str, Dict[Any, int]],  # Loaded by category_mapping_utils.load_mappings
        handle_unknown_value: str = "use_mapping_default"  # or 'assign_new_code' or 'error'
) -> pd.DataFrame:
    """
    Encodes categorical columns using provided mappings.

    Args:
        df: Input Pandas DataFrame.
        categorical_cols: List of categorical column names to process.
        category_mappings: A dictionary where keys are column names, and values are
                           dictionaries mapping original category values to integer codes.
                           Example: {'protocol': {'TCP': 0, 'UDP': 1, 'unknown': 2}}
        handle_unknown_value: Strategy for values not in mapping.
            'use_mapping_default': Tries to use mapping[repr("unknown")] or mapping["unknown"].
                                   If "unknown" itself is not in the mapping, it's an issue.
            (Future options: 'assign_new_code', 'error')


    Returns:
        DataFrame with encoded categorical columns (as integers).
    """
    logging.info(f"Encoding {len(categorical_cols)} categorical features...")
    df_processed = df.copy()

    for col in categorical_cols:
        if col not in df_processed.columns:
            logging.warning(f"Categorical column '{col}' not found in DataFrame. Skipping.")
            continue

        # Ensure column is treated as string for consistent mapping, especially if mixed types exist
        # NaNs should be filled before this step if "unknown" string is to be mapped.
        df_processed[col] = df_processed[col].astype(str)  # Convert to string to match repr keys if needed

        if col not in category_mappings:
            logging.warning(f"No category mapping found for column '{col}'. "
                            "Attempting to encode using pd.Categorical.codes (on-the-fly, may be inconsistent across calls/datasets).")
            df_processed[col] = pd.Categorical(df_processed[col]).codes
            continue

        mapping_for_col = category_mappings[col]

        # Determine the code for "unknown" values from the specific column's mapping
        unknown_code = None
        # The keys in mapping_for_col are original values (e.g., string "unknown", int 0)
        # as ast.literal_eval was used during loading.
        if "unknown" in mapping_for_col:  # Check for actual string "unknown"
            unknown_code = mapping_for_col["unknown"]
        elif repr("unknown") in mapping_for_col:  # Check for repr("unknown") if keys were stored that way consistently
            unknown_code = mapping_for_col[repr("unknown")]

        if unknown_code is None:
            # This is a problem if "unknown" is expected to handle NaNs/new values
            logging.warning(f"No explicit 'unknown' category code found in mapping for column '{col}'. "
                            "Unseen values might lead to errors or be mapped to NaN by .map(). "
                            "Consider adding 'unknown' to your mappings or using a default fill for unmappable values.")
            # Fallback: if a value is not in the map, .map will produce NaN, then astype(int) would fail.
            # We should handle this by filling NaNs from .map() with a default code.
            # For now, let's assume if unknown_code is None, we default to 0 for unmapped values.
            unknown_code = 0  # Default if "unknown" not explicitly in map.
            logging.warning(f"Defaulting unmappable values in '{col}' to code {unknown_code} as 'unknown' not in map.")

        # Apply the mapping. Values not in the mapping will become NaN by .map()
        # Then fill these NaNs with the determined unknown_code.
        # The keys in mapping_for_col are the original values.
        encoded_series = df_processed[col].map(mapping_for_col)
        encoded_series.fillna(unknown_code, inplace=True)

        df_processed[col] = encoded_series.astype(np.int64)  # Ensure integer type

    return df_processed


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (fp_test) %(message)s')

    # --- Create Dummy Data and Configs for Testing ---
    sample_df_data = {
        'duration': [10.0, 20.5, np.nan, 5.0, 1000.0],
        'packets': [1, 5, 3, 888, 6],
        'protocol': ['TCP', 'UDP', 'TCP', 'ICMP', 'UDP'],
        'service': ['http', 'dns', 'http', 'other', 'unknown'],
        'other_cat': [100, 200, 100, 300, 200]
    }
    sample_df = pd.DataFrame(sample_df_data)

    test_numerical_cols = ['duration', 'packets']
    test_categorical_cols = ['protocol', 'service', 'other_cat']

    # Dummy Standardization Stats (normally loaded from .npz)
    test_stats = {
        "cols": test_numerical_cols,  # Important: cols for which stats are defined
        "mean": {'duration': 50.0, 'packets': 100.0},
        "std": {'duration': 20.0, 'packets': 50.0},
        "median": {'duration': 15.0, 'packets': 5.0},  # Medians used for NaN filling
        "clip_low": {'duration': 0.0, 'packets': 0.0},
        "clip_high": {'duration': 500.0, 'packets': 500.0}
    }
    # Save dummy stats to a temp .npz file for load_standardization_stats test
    temp_stats_path = "temp_test_stats.npz"
    np.savez(temp_stats_path,
             cols=np.array(test_stats["cols"]),
             mean=np.array([test_stats["mean"][col] for col in test_stats["cols"]]),
             std=np.array([test_stats["std"][col] for col in test_stats["cols"]]),
             median=np.array([test_stats["median"][col] for col in test_stats["cols"]]),
             clip_low=np.array([test_stats["clip_low"][col] for col in test_stats["cols"]]),
             clip_high=np.array([test_stats["clip_high"][col] for col in test_stats["cols"]]))

    # Dummy Category Mappings (normally loaded from JSON by category_mapping_utils)
    # Keys are original values, values are integer codes.
    test_mappings = {
        'protocol': {'TCP': 0, 'UDP': 1, 'ICMP': 2, "unknown": 3},  # "unknown" string as a category
        'service': {'http': 0, 'dns': 1, 'other': 2, "unknown": 3},
        'other_cat': {100: 0, 200: 1, 300: 2, "unknown": 3}  # Example with integer categories
    }

    print("\n--- Testing load_standardization_stats ---")
    loaded_stats = load_standardization_stats(temp_stats_path)
    if loaded_stats:
        print("Successfully loaded stats.")
        # print(loaded_stats)
        assert loaded_stats["cols"] == test_stats["cols"]
        assert loaded_stats["mean"]['duration'] == test_stats["mean"]['duration']
    else:
        print("Failed to load stats.")

    print("\n--- Testing process_numerical_features ---")
    df_num_processed = process_numerical_features(sample_df.copy(), test_numerical_cols, test_stats)
    print("Numerically processed DataFrame (head):\n", df_num_processed[test_numerical_cols].head())
    # Check if NaN in 'duration' was filled and then standardized
    # Original NaN was at index 2. Median for duration is 15.0.
    # Standardized: (15.0 - 50.0) / 20.0 = -35.0 / 20.0 = -1.75
    expected_duration_idx2 = (test_stats["median"]['duration'] - test_stats["mean"]['duration']) / (
                test_stats["std"]['duration'] + 1e-6)
    print(
        f"Value for duration at index 2 (original NaN): {df_num_processed.loc[2, 'duration']:.4f} (Expected around: {expected_duration_idx2:.4f})")
    assert np.isclose(df_num_processed.loc[2, 'duration'], expected_duration_idx2)
    # Check clipping: 'packets' at index 3 was 888, clip_high is 500.
    # Standardized: (500.0 - 100.0) / 50.0 = 400.0 / 50.0 = 8.0
    expected_packets_idx3 = (test_stats["clip_high"]['packets'] - test_stats["mean"]['packets']) / (
                test_stats["std"]['packets'] + 1e-6)
    print(
        f"Value for packets at index 3 (original 888, clipped to 500): {df_num_processed.loc[3, 'packets']:.4f} (Expected: {expected_packets_idx3:.4f})")
    assert np.isclose(df_num_processed.loc[3, 'packets'], expected_packets_idx3)

    print("\n--- Testing encode_categorical_features ---")
    # First, fill NaNs in categorical columns of the original sample_df with "unknown" string
    # This simulates what the main pipeline would do before calling encode_categorical_features
    sample_df_cat_filled = sample_df.copy()
    for col in test_categorical_cols:
        sample_df_cat_filled[col] = sample_df_cat_filled[col].fillna("unknown").astype(str)
        # For 'other_cat' which has ints, ensure "unknown" is handled if it was NaN
        if col == 'other_cat':  # If it had actual NaNs, they become string 'nan'
            sample_df_cat_filled[col] = sample_df_cat_filled[col].replace('nan', 'unknown')

    df_cat_processed = encode_categorical_features(sample_df_cat_filled, test_categorical_cols, test_mappings)
    print("Categorically encoded DataFrame (head):\n", df_cat_processed[test_categorical_cols].head())
    # Expected for 'protocol' at index 0 ('TCP') is 0
    assert df_cat_processed.loc[0, 'protocol'] == test_mappings['protocol']['TCP']
    # Expected for 'service' at index 4 ('unknown') is 3
    assert df_cat_processed.loc[4, 'service'] == test_mappings['service']['unknown']
    # Expected for 'other_cat' at index 0 (original 100) is 0
    # The mapping keys are original values, so we map string '100'
    assert df_cat_processed.loc[0, 'other_cat'] == test_mappings['other_cat'][100]  # Original key was int

    print("\n✅ All tests seemed to pass based on output.")

    # Clean up
    if os.path.exists(temp_stats_path):
        os.remove(temp_stats_path)

f