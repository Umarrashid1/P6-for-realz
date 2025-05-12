# pipeline/flow_processor.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging


# Configure basic logging for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (flow_processor) %(message)s')

def prepare_flow_data_for_tensor_conversion(
        processed_df: pd.DataFrame,
        labels: List[int],  # List of integer labels, same length as df
        numerical_cols: List[str],
        categorical_cols: List[str]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepares numerical, categorical, and label data from a processed DataFrame
    into NumPy arrays, ready for conversion to PyTorch tensors.
    Assumes the input DataFrame has numerical features already scaled/standardized
    and categorical features already integer-encoded.

    Args:
        processed_df: Pandas DataFrame where each row is a flow and features are processed.
        labels: A list of integer labels corresponding to each row in processed_df.
        numerical_cols: List of numerical column names to extract.
        categorical_cols: List of categorical column names to extract (should contain integer codes).

    Returns:
        A tuple of NumPy arrays: (numerical_data, categorical_data, label_data)
        Returns None if there's a mismatch in lengths or essential columns are missing.
    """
    logging.info(
        f"Preparing flow data for tensor conversion. Input df shape: {processed_df.shape}, Num labels: {len(labels)}")

    if len(processed_df) != len(labels):
        logging.error(f"DataFrame length ({len(processed_df)}) and labels length ({len(labels)}) mismatch.")
        return None

    if processed_df.empty:
        logging.warning("Input DataFrame is empty. Returning None.")
        return None

    # --- Extract Numerical Features ---
    missing_num_cols = [col for col in numerical_cols if col not in processed_df.columns]
    if missing_num_cols:
        logging.error(f"Missing numerical columns in DataFrame: {missing_num_cols}")
        # Depending on strictness, you might return None or proceed with available columns
        # For now, let's be strict.
        return None

    try:
        if numerical_cols:
            numerical_data_np = processed_df[numerical_cols].values.astype(np.float32)
        else:
            # Create an empty array with the correct first dimension if no numerical columns
            numerical_data_np = np.empty((len(processed_df), 0), dtype=np.float32)
        logging.debug(f"Extracted numerical data shape: {numerical_data_np.shape}")
    except Exception as e:
        logging.error(f"Error extracting numerical data: {e}")
        return None

    # --- Extract Categorical Features ---
    missing_cat_cols = [col for col in categorical_cols if col not in processed_df.columns]
    if missing_cat_cols:
        logging.error(f"Missing categorical columns in DataFrame: {missing_cat_cols}")
        return None

    try:
        if categorical_cols:
            # Ensure these columns are indeed integer-encoded
            # For safety, explicitly cast to int64, though they should be from previous step
            categorical_data_np = processed_df[categorical_cols].values.astype(np.int64)
        else:
            # Create an empty array with the correct first dimension if no categorical columns
            categorical_data_np = np.empty((len(processed_df), 0), dtype=np.int64)
        logging.debug(f"Extracted categorical data shape: {categorical_data_np.shape}")
    except Exception as e:
        logging.error(f"Error extracting categorical data: {e}")
        return None

    # --- Prepare Labels ---
    try:
        label_data_np = np.array(labels, dtype=np.int64)
        logging.debug(f"Prepared label data shape: {label_data_np.shape}")
    except Exception as e:
        logging.error(f"Error preparing label data: {e}")
        return None

    logging.info("Successfully prepared numerical, categorical, and label NumPy arrays.")
    return numerical_data_np, categorical_data_np, label_data_np


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] (fp_test) %(message)s')

    # --- Create Dummy Data for Testing ---
    sample_flow_data = {
        'flow_duration_norm': [0.1, -0.5, 1.2, 0.0],
        'total_packets_norm': [-1.0, 0.5, 0.0, 2.1],
        'protocol_code': [0, 1, 0, 2],  # Already integer encoded
        'service_code': [1, 0, 1, 2],  # Already integer encoded
        # 'extra_num_col': [0.5, 0.6, 0.7, 0.8] # Test with an extra col not in list
    }
    sample_labels = [0, 1, 0, 1]
    sample_df = pd.DataFrame(sample_flow_data)

    test_numerical_cols = ['flow_duration_norm', 'total_packets_norm']
    test_categorical_cols = ['protocol_code', 'service_code']

    print("\n--- Testing prepare_flow_data_for_tensor_conversion ---")
    result = prepare_flow_data_for_tensor_conversion(
        sample_df,
        sample_labels,
        test_numerical_cols,
        test_categorical_cols
    )

    if result:
        num_arr, cat_arr, lbl_arr = result
        print("\nShapes of returned NumPy arrays:")
        print(f"  Numerical: {num_arr.shape}, dtype: {num_arr.dtype}")
        print(f"  Categorical: {cat_arr.shape}, dtype: {cat_arr.dtype}")
        print(f"  Labels: {lbl_arr.shape}, dtype: {lbl_arr.dtype}")

        print("\nContent (first 2 rows):")
        print("Numerical:\n", num_arr[:2])
        print("Categorical:\n", cat_arr[:2])
        print("Labels:\n", lbl_arr[:2])

        # Basic assertions
        assert num_arr.shape == (4, 2)
        assert cat_arr.shape == (4, 2)
        assert lbl_arr.shape == (4,)
        assert num_arr.dtype == np.float32
        assert cat_arr.dtype == np.int64
        assert lbl_arr.dtype == np.int64
        print("\n✅ Basic tests passed.")
    else:
        print("❌ Test failed: Result was None.")

    print("\n--- Test with missing numerical column ---")
    result_missing_num = prepare_flow_data_for_tensor_conversion(
        sample_df, sample_labels, ['non_existent_num'] + test_numerical_cols, test_categorical_cols
    )
    assert result_missing_num is None, "Test failed: Should return None for missing numerical column."
    print("✅ Correctly returned None for missing numerical column.")

    print("\n--- Test with empty numerical columns list ---")
    result_empty_num = prepare_flow_data_for_tensor_conversion(
        sample_df, sample_labels, [], test_categorical_cols
    )
    if result_empty_num:
        num_arr, _, _ = result_empty_num
        assert num_arr.shape == (4, 0), "Test failed: Numerical array shape incorrect for empty num_cols."
        print("✅ Correctly handled empty numerical_cols list.")
    else:
        print("❌ Test failed for empty numerical_cols list.")

    print("\n--- Test with empty DataFrame ---")
    empty_df = pd.DataFrame(columns=sample_df.columns)
    result_empty_df = prepare_flow_data_for_tensor_conversion(
        empty_df, [], test_numerical_cols, test_categorical_cols
    )
    assert result_empty_df is None, "Test failed: Should return None for empty DataFrame."
    print("✅ Correctly returned None for empty DataFrame.")

    print("\n--- Test with label length mismatch ---")
    result_label_mismatch = prepare_flow_data_for_tensor_conversion(
        sample_df, [0, 1], test_numerical_cols, test_categorical_cols
    )
    assert result_label_mismatch is None, "Test failed: Should return None for label mismatch."
    print("✅ Correctly returned None for label length mismatch.")

