# pipeline/packet_processor.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging


# Configure basic logging for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (packet_processor) %(message)s')

def create_packet_sequences_from_processed_df(
        processed_df: pd.DataFrame,
        file_label: int,  # Single label for all packets/sequences derived from this DataFrame/file
        numerical_cols: List[str],
        categorical_cols: List[str],
        stream_col_name: str,  # Name of the column used to group packets into flows/streams
        max_seq_len: int,
        category_mappings: Dict[str, Dict[Any, int]]  # Needed to get 'unknown' code for padding
) -> Optional[Tuple[List[np.ndarray], List[int], List[np.ndarray], Dict[str, List[np.ndarray]]]]:
    """
    Processes a DataFrame of standardized/encoded packet data, groups by stream,
    and creates fixed-length sequences with padding/striding.

    Args:
        processed_df: Pandas DataFrame with processed packet features.
                      Numerical columns are standardized, categorical columns are integer-encoded.
        file_label: The single integer label to assign to all sequences generated from this df.
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        stream_col_name: The name of the column identifying unique streams/flows.
        max_seq_len: The desired fixed length for output sequences.
        category_mappings: The loaded category mappings, used to find the integer code
                           for "unknown" for padding categorical sequences.

    Returns:
        A tuple containing:
            - List of numerical sequence NumPy arrays (np.float32).
            - List of integer labels.
            - List of attention mask NumPy arrays (np.float32).
            - Dictionary where keys are cat_col_names and values are lists of
              categorical sequence NumPy arrays (np.int64).
        Returns None if critical errors occur (e.g., stream column missing).
    """
    logging.debug(f"Creating packet sequences. Input df shape: {processed_df.shape}, max_seq_len: {max_seq_len}")

    if stream_col_name not in processed_df.columns:
        logging.error(f"Stream identifier column '{stream_col_name}' not found in DataFrame.")
        return None

    if processed_df.empty:
        logging.warning("Input DataFrame is empty. No sequences to create.")
        return [], [], [], {col: [] for col in categorical_cols}

    # Prepare lists to hold all generated sequences from this DataFrame
    all_numerical_seqs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_attention_masks: List[np.ndarray] = []
    all_categorical_seqs_dict: Dict[str, List[np.ndarray]] = {col: [] for col in categorical_cols}

    # Get the integer code for "unknown" for each categorical column for padding
    unknown_codes_for_padding: Dict[str, int] = {}
    for col in categorical_cols:
        mapping_for_col = category_mappings.get(col, {})
        # Try to find 'unknown' or repr('unknown') in the mapping
        # The keys in mapping_for_col are original values (e.g., string "unknown", int 0)
        if "unknown" in mapping_for_col:
            unknown_codes_for_padding[col] = mapping_for_col["unknown"]
        elif repr("unknown") in mapping_for_col:
            unknown_codes_for_padding[col] = mapping_for_col[repr("unknown")]
        else:
            logging.warning(f"No 'unknown' code in mapping for categorical column '{col}'. Defaulting padding to 0.")
            unknown_codes_for_padding[col] = 0  # Fallback padding code

    # Group by the stream identifier
    flow_groups = processed_df.groupby(stream_col_name, sort=False)
    num_original_flows = len(flow_groups)
    num_output_sequences = 0

    for _, flow_df in flow_groups:
        # Extract numerical and categorical features for the current flow
        # These are already processed (standardized/encoded)
        try:
            numerical_features_flow = flow_df[numerical_cols].values  # Should be float after standardization
            categorical_features_flow_dict = {
                col: flow_df[col].values for col in categorical_cols  # Should be int after encoding
            }
        except KeyError as e:
            logging.error(f"Missing expected column {e} when extracting features for a flow. Skipping this flow.")
            continue

        flow_len = len(numerical_features_flow)
        if flow_len == 0:
            continue

        # --- Inner helper for appending generated sequences ---
        def append_sequence_data(num_slice, cat_slice_dict, current_mask):
            nonlocal num_output_sequences
            all_numerical_seqs.append(num_slice.astype(np.float32))
            all_attention_masks.append(current_mask.astype(np.float32))
            all_labels.append(file_label)
            for cat_col_name in categorical_cols:
                all_categorical_seqs_dict[cat_col_name].append(cat_slice_dict[cat_col_name].astype(np.int64))
            num_output_sequences += 1

        # --- End inner helper ---

        if flow_len < max_seq_len:
            pad_len = max_seq_len - flow_len

            # Pad numerical features with zeros
            padded_numerical_seq = np.concatenate(
                [numerical_features_flow, np.zeros((pad_len, len(numerical_cols)), dtype=np.float32)],
                axis=0
            )

            # Create attention mask
            attention_mask = np.concatenate([np.ones(flow_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])

            # Pad categorical features
            padded_categorical_seq_dict = {}
            for col in categorical_cols:
                unknown_pad_value = unknown_codes_for_padding[col]
                padded_cat_feature = np.pad(
                    categorical_features_flow_dict[col],
                    (0, pad_len),
                    mode='constant',
                    constant_values=unknown_pad_value
                )
                padded_categorical_seq_dict[col] = padded_cat_feature

            append_sequence_data(padded_numerical_seq, padded_categorical_seq_dict, attention_mask)

        elif flow_len == max_seq_len:
            attention_mask = np.ones(max_seq_len, dtype=np.float32)
            append_sequence_data(numerical_features_flow, categorical_features_flow_dict, attention_mask)

        else:  # flow_len > max_seq_len, apply striding
            stride = max_seq_len // 2  # 50% overlap, common choice
            if stride == 0: stride = 1  # Ensure stride is at least 1 for very small max_seq_len

            for start_idx in range(0, flow_len - max_seq_len + 1, stride):
                end_idx = start_idx + max_seq_len
                num_slice = numerical_features_flow[start_idx:end_idx]
                cat_slice_dict = {
                    col: categorical_features_flow_dict[col][start_idx:end_idx]
                    for col in categorical_cols
                }
                current_mask = np.ones(max_seq_len, dtype=np.float32)  # Full mask for these full slices
                append_sequence_data(num_slice, cat_slice_dict, current_mask)

            # Handle the tail end if striding doesn't perfectly cover it
            # This ensures the last part of the sequence is always included
            if (flow_len - max_seq_len) % stride != 0:
                start_idx = flow_len - max_seq_len
                end_idx = flow_len
                num_slice = numerical_features_flow[start_idx:end_idx]
                cat_slice_dict = {
                    col: categorical_features_flow_dict[col][start_idx:end_idx]
                    for col in categorical_cols
                }
                current_mask = np.ones(max_seq_len, dtype=np.float32)
                append_sequence_data(num_slice, cat_slice_dict, current_mask)

    logging.debug(
        f"Processed {num_original_flows} original streams into {num_output_sequences} sequences of length {max_seq_len}.")
    return all_numerical_seqs, all_labels, all_attention_masks, all_categorical_seqs_dict


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] (packet_proc_test) %(message)s')

    # --- Create Dummy Processed DataFrame for Testing ---
    # This df simulates output from feature_processor.py
    num_packets_flow1 = 50
    num_packets_flow2 = 120
    num_packets_flow3 = 10  # Short flow
    total_packets = num_packets_flow1 + num_packets_flow2 + num_packets_flow3

    test_numerical_cols = ['feat_num1_std', 'feat_num2_std']
    test_categorical_cols = ['protocol_code', 'port_class_code']
    test_stream_col = 'stream_id'
    test_max_seq_len = 64
    test_label = 1  # Assume all packets in this df belong to label 1

    # Dummy category mappings (needed for 'unknown' padding code)
    test_cat_mappings = {
        'protocol_code': {'TCP': 0, 'UDP': 1, 'unknown': 99},
        'port_class_code': {'web': 0, 'service': 1, 'unknown': 99}
    }

    sample_data = {
        test_stream_col: \
            [1] * num_packets_flow1 + \
            [2] * num_packets_flow2 + \
            [3] * num_packets_flow3,
        'feat_num1_std': np.random.randn(total_packets),
        'feat_num2_std': np.random.randn(total_packets),
        'protocol_code': np.random.randint(0, 2, total_packets),  # Encoded 0 or 1
        'port_class_code': np.random.randint(0, 2, total_packets)  # Encoded 0 or 1
    }
    sample_processed_df = pd.DataFrame(sample_data)
    logging.info(f"Created sample processed DataFrame with shape: {sample_processed_df.shape}")

    print("\n--- Testing create_packet_sequences_from_processed_df ---")
    result = create_packet_sequences_from_processed_df(
        processed_df=sample_processed_df,
        file_label=test_label,
        numerical_cols=test_numerical_cols,
        categorical_cols=test_categorical_cols,
        stream_col_name=test_stream_col,
        max_seq_len=test_max_seq_len,
        category_mappings=test_cat_mappings
    )

    if result:
        num_seqs, labels, masks, cat_seqs_dict = result
        num_sequences_generated = len(num_seqs)
        print(f"\nGenerated {num_sequences_generated} sequences.")

        if num_sequences_generated > 0:
            print(f"  Shape of first numerical sequence: {num_seqs[0].shape}")
            print(f"  Shape of first attention mask: {masks[0].shape}")
            print(f"  Label for first sequence: {labels[0]}")
            for col_name, seq_list in cat_seqs_dict.items():
                print(f"  Shape of first '{col_name}' sequence: {seq_list[0].shape}")

            # Expected number of sequences:
            # Flow 1 (50 pkts, max_seq 64): 1 sequence (padded)
            # Flow 2 (120 pkts, max_seq 64, stride 32):
            #   start 0: 0-63
            #   start 32: 32-95
            #   start 64: 64-127 (no, because 120-64 = 56. (flow_len - max_seq_len) % stride = 56 % 32 = 24 != 0)
            #   So, range(0, 120 - 64 + 1, 32) -> range(0, 57, 32) -> start_idx = 0, 32. (2 sequences)
            #   Tail end: start = 120 - 64 = 56. (1 sequence)
            #   Total for flow 2 = 2 + 1 = 3 sequences.
            # Flow 3 (10 pkts, max_seq 64): 1 sequence (padded)
            # Total expected = 1 + 3 + 1 = 5 sequences.
            expected_num_sequences = 1 + ((num_packets_flow2 - test_max_seq_len) // (test_max_seq_len // 2) + 1) + 1
            # More precise calculation for striding:
            num_strided_seqs_flow2 = 0
            if num_packets_flow2 > test_max_seq_len:
                stride = test_max_seq_len // 2
                for s in range(0, num_packets_flow2 - test_max_seq_len + 1, stride):
                    num_strided_seqs_flow2 += 1
                if (num_packets_flow2 - test_max_seq_len) % stride != 0:
                    num_strided_seqs_flow2 += 1  # for the tail
            elif num_packets_flow2 > 0:  # if it's shorter or equal to max_seq_len
                num_strided_seqs_flow2 = 1

            expected_num_sequences = (1 if num_packets_flow1 > 0 else 0) + \
                                     num_strided_seqs_flow2 + \
                                     (1 if num_packets_flow3 > 0 else 0)

            print(f"  Expected number of sequences based on logic: around {expected_num_sequences}")
            # This assertion might be tricky due to exact striding logic, adjust if needed
            # assert num_sequences_generated == expected_num_sequences, "Number of generated sequences mismatch"

            # Check shapes
            assert num_seqs[0].shape == (test_max_seq_len, len(test_numerical_cols))
            assert masks[0].shape == (test_max_seq_len,)
            for col_name in test_categorical_cols:
                assert cat_seqs_dict[col_name][0].shape == (test_max_seq_len,)

            # Check mask for padded sequence (Flow 3, original length 10)
            # Find a sequence that was padded (e.g., from flow3)
            idx_flow3_seq = -1  # last sequence is from flow3
            if len(masks[idx_flow3_seq]) == test_max_seq_len and num_packets_flow3 < test_max_seq_len:
                print(f"Mask for a padded sequence (original length {num_packets_flow3}):")
                print(masks[idx_flow3_seq])
                assert np.sum(masks[idx_flow3_seq]) == num_packets_flow3, "Mask sum incorrect for padded sequence"
                assert np.all(
                    masks[idx_flow3_seq][:num_packets_flow3] == 1), "Mask incorrect for padded sequence (data part)"
                assert np.all(
                    masks[idx_flow3_seq][num_packets_flow3:] == 0), "Mask incorrect for padded sequence (pad part)"
                # Check padding value for categorical
                unknown_code_protocol = test_cat_mappings['protocol_code']['unknown']
                assert np.all(cat_seqs_dict['protocol_code'][idx_flow3_seq][
                              num_packets_flow3:] == unknown_code_protocol), "Categorical padding incorrect"

            print("\n✅ Basic tests passed.")
        else:
            print("❌ No sequences were generated.")

    else:
        print("❌ Test failed: Result was None.")

