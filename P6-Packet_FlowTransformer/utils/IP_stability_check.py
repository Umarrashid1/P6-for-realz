import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import torch
import json
import ast # For literal_eval when loading mappings

# --- Configuration ---
# **** Paths to the preprocessed data and mappings ****
PREPROCESSED_PT_FILE = '../../dataset/packet_small.pt' # CHANGE THIS to your .pt file generated from PACKET data
MAPPINGS_JSON_FILE = 'category_mappings.json' # CHANGE THIS if your mapping file has a different name/path

# --- Import column names from pipeline config ---
# We need this to know which columns in the categorical tensor correspond to IP/MAC
try:
    from pipeline.config import categorical_columns as packet_categorical_columns
    print("✅ Successfully imported column definitions from pipeline.config")

    EXPECTED_SRC_IP_COL = 'src_ip'
    EXPECTED_DST_IP_COL = 'dst_ip'
    EXPECTED_SRC_MAC_COL = 'src_mac'
    EXPECTED_DST_MAC_COL = 'dst_mac'

    # Find the indices of these columns in the config list
    # This assumes the order in the .pt file's categorical tensor matches this list
    try:
        SRC_IP_IDX = packet_categorical_columns.index(EXPECTED_SRC_IP_COL)
        DST_IP_IDX = packet_categorical_columns.index(EXPECTED_DST_IP_COL)
        SRC_MAC_IDX = packet_categorical_columns.index(EXPECTED_SRC_MAC_COL)
        DST_MAC_IDX = packet_categorical_columns.index(EXPECTED_DST_MAC_COL)
        print(f"   Index mapping: src_ip={SRC_IP_IDX}, dst_ip={DST_IP_IDX}, src_mac={SRC_MAC_IDX}, dst_mac={DST_MAC_IDX}")
        IP_COL_INDICES = [SRC_IP_IDX, DST_IP_IDX]
        MAC_COL_INDICES = [SRC_MAC_IDX, DST_MAC_IDX]
        IP_COL_NAMES = [EXPECTED_SRC_IP_COL, EXPECTED_DST_IP_COL]
        MAC_COL_NAMES = [EXPECTED_SRC_MAC_COL, EXPECTED_DST_MAC_COL]

    except ValueError as e:
        print(f"❌ Error: Column '{e.args[0].split()[0]}' not found in imported config.categorical_columns.")
        print("   Cannot determine column indices. Exiting.")
        exit(1)

except ImportError:
    print("⚠️ Warning: Could not import from 'pipeline.config'.")
    print("   Cannot determine column indices automatically. Exiting.")
    exit(1)
except AttributeError:
     print("⚠️ Warning: 'categorical_columns' not found in imported 'pipeline.config'.")
     print("   Cannot determine column indices automatically. Exiting.")
     exit(1)


def load_mappings_from_json(path="category_mappings.json") -> Dict[str, Dict[Any, int]]:
    """Loads mappings, converting string representations back to original types."""
    print(f"Loading category mappings from {path}...")
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        # Convert repr'd keys back to original types (int, float, str, etc.)
        # Handle potential errors during literal_eval
        loaded_mappings = {}
        for col, mapping in raw.items():
            converted_mapping = {}
            for k_repr, v_int in mapping.items():
                try:
                    original_key = ast.literal_eval(k_repr)
                    converted_mapping[original_key] = v_int
                except (ValueError, SyntaxError, TypeError) as e:
                    print(f"  [Warning] Could not evaluate key '{k_repr}' for column '{col}'. Skipping. Error: {e}")
                    continue # Skip problematic keys
            loaded_mappings[col] = converted_mapping
        print("Mappings loaded successfully.")
        return loaded_mappings
    except FileNotFoundError:
        print(f"❌ Error: Mappings file not found at {path}")
        raise
    except json.JSONDecodeError as e:
         print(f"❌ Error: Could not decode JSON from {path}: {e}")
         raise
    except Exception as e:
        print(f"❌ Error loading or processing mappings file {path}: {e}")
        raise


def create_reverse_mappings(mappings: Dict[str, Dict[Any, int]]) -> Dict[str, Dict[int, Any]]:
    """Creates reverse mappings (code -> original value)."""
    reverse_mappings = {}
    print("Creating reverse mappings (code -> original value)...")
    for col, mapping in mappings.items():
        reverse_map = {v_int: k_orig for k_orig, v_int in mapping.items()}
        if len(reverse_map) != len(mapping):
             print(f"  [Warning] Potential duplicate codes detected in mapping for column '{col}'. Reverse map might be incomplete.")
        reverse_mappings[col] = reverse_map
    return reverse_mappings


def check_ip_mac_stability_from_pt(
    pt_file_path: str,
    mappings_json_path: str,
    ip_col_indices: List[int],
    mac_col_indices: List[int],
    ip_col_names: List[str], # Original names for mapping lookup
    mac_col_names: List[str] # Original names for mapping lookup
    ) -> Tuple[str, pd.DataFrame]:
    """
    Analyzes preprocessed data (.pt file) using category mappings (.json)
    to determine IP-MAC stability.

    Args:
        pt_file_path: Path to the .pt file containing 'categorical' tensor.
        mappings_json_path: Path to the category_mappings.json file.
        ip_col_indices: List of column indices for IP addresses in the tensor.
        mac_col_indices: List of column indices for MAC addresses in the tensor.
        ip_col_names: List of original IP column names corresponding to indices.
        mac_col_names: List of original MAC column names corresponding to indices.


    Returns:
        A tuple containing:
          - Recommended strategy ('A' for static, 'B' for dynamic).
          - A DataFrame of unique IP (string) - MAC (string) pairs found.
    """
    print(f"\n--- Starting IP-MAC Stability Check (using .pt file) ---")
    print(f"PT File: {pt_file_path}")
    print(f"Mappings File: {mappings_json_path}")

    # 1. Load Mappings and Create Reverse Mappings
    try:
        mappings = load_mappings_from_json(mappings_json_path)
        reverse_mappings = create_reverse_mappings(mappings)
    except Exception as e:
        print(f"❌ Failed to load or process mappings. Cannot proceed.")
        raise # Re-raise the exception

    # 2. Load Preprocessed Data
    print(f"\n1. Loading preprocessed data from {pt_file_path}...")
    try:
        data = torch.load(pt_file_path, map_location='cpu') # Load to CPU
        if 'categorical' not in data:
            raise ValueError("Key 'categorical' not found in the .pt file.")
        categorical_tensor = data['categorical']
        num_samples, num_features = categorical_tensor.shape
        print(f"   Loaded 'categorical' tensor with shape: {categorical_tensor.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Preprocessed data file not found at {pt_file_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")
        raise

    # 3. Reconstruct IP/MAC Strings
    print("\n2. Reconstructing IP and MAC strings from codes...")
    all_pairs_list = []
    required_indices = list(set(ip_col_indices + mac_col_indices))

    # Check if indices are valid
    if any(idx >= num_features for idx in required_indices):
         raise ValueError(f"One or more required column indices ({required_indices}) are out of bounds for the tensor shape {categorical_tensor.shape}")

    # Extract relevant columns from tensor
    codes_df = pd.DataFrame(categorical_tensor[:, required_indices].numpy(),
                            columns=[packet_categorical_columns[i] for i in required_indices])

    for ip_idx, mac_idx, ip_name, mac_name in zip(ip_col_indices, mac_col_indices, ip_col_names, mac_col_names):
        print(f"   Processing pair: {ip_name} (idx {ip_idx}) - {mac_name} (idx {mac_idx})")
        # Get the reverse maps for the specific IP and MAC columns
        try:
            reverse_map_ip = reverse_mappings[ip_name]
            reverse_map_mac = reverse_mappings[mac_name]
        except KeyError as e:
             print(f"❌ Error: Column name '{e.args[0]}' not found in loaded mappings. Check config and mapping file consistency.")
             raise

        # Get the integer codes from the DataFrame using original names
        ip_codes = codes_df[ip_name]
        mac_codes = codes_df[mac_name]

        # Map codes back to strings - use .get() for safety against missing codes
        ip_strings = ip_codes.map(lambda code: reverse_map_ip.get(code, np.nan))
        mac_strings = mac_codes.map(lambda code: reverse_map_mac.get(code, np.nan))

        pairs = pd.DataFrame({'IP': ip_strings, 'MAC': mac_strings})
        all_pairs_list.append(pairs)

    if not all_pairs_list:
        raise RuntimeError("Failed to reconstruct any IP-MAC pairs.")

    print("\n3. Aggregating and cleaning reconstructed pairs...")
    all_pairs = pd.concat(all_pairs_list, ignore_index=True)
    print(f"   Combined pairs count (raw reconstructed): {len(all_pairs)}")

    # Clean reconstructed strings (handle NaNs introduced by missing codes)
    initial_rows = len(all_pairs)
    all_pairs.dropna(subset=['IP', 'MAC'], inplace=True)
    # Convert to string type *after* dropna to avoid issues with NaN comparison
    all_pairs['IP'] = all_pairs['IP'].astype(str)
    all_pairs['MAC'] = all_pairs['MAC'].astype(str)
    rows_after_nan_drop = len(all_pairs)
    print(f"   Pairs count after dropping NaNs: {rows_after_nan_drop} (Removed {initial_rows - rows_after_nan_drop})")


    # Drop duplicate IP-MAC pairs
    unique_ip_mac_pairs = all_pairs.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(unique_ip_mac_pairs)
    print(f"   Unique IP-MAC pairs count: {rows_after_dedup} (Removed {rows_after_nan_drop - rows_after_dedup} duplicates)")

    if unique_ip_mac_pairs.empty:
         raise RuntimeError("No valid unique IP-MAC pairs found after reconstruction and cleaning.")

    print("\n4. Analyzing stability...")
    # Group by IP address and count the number of unique MAC addresses
    ip_stability_counts = unique_ip_mac_pairs.groupby('IP')['MAC'].nunique()

    # Identify IPs associated with more than one MAC address
    multi_mac_ips = ip_stability_counts[ip_stability_counts > 1]

    # (Rest of the reporting and conclusion logic is identical to the previous script)
    # 5. Reporting Findings
    total_unique_ips = len(ip_stability_counts)
    num_multi_mac_ips = len(multi_mac_ips)
    num_static_ips = total_unique_ips - num_multi_mac_ips

    print("\n--- Stability Results ---")
    print(f"Total unique IP addresses analyzed: {total_unique_ips}")
    print(f"Number of IPs mapped to only ONE unique MAC address: {num_static_ips}")
    print(f"Number of IPs mapped to MORE THAN ONE unique MAC address: {num_multi_mac_ips}")

    if num_multi_mac_ips > 0:
        print(f"\nIPs associated with multiple MAC addresses ({num_multi_mac_ips} total):")
        max_examples_to_print = 20
        print(multi_mac_ips.sort_values(ascending=False).head(max_examples_to_print))
        if num_multi_mac_ips > max_examples_to_print:
            print(f"... and {num_multi_mac_ips - max_examples_to_print} more.")
        print("\nExample details for unstable IPs:")
        for ip_addr in multi_mac_ips.sort_values(ascending=False).head(min(5, num_multi_mac_ips)).index:
             associated_macs = unique_ip_mac_pairs[unique_ip_mac_pairs['IP'] == ip_addr]['MAC'].unique()
             print(f"  - IP: {ip_addr} maps to {len(associated_macs)} MACs: {list(associated_macs)}")
    else:
        print("\nAll analyzed IP addresses map to only one unique MAC address.")

    # 6. Conclusion & Recommendation
    print("\n--- Conclusion ---")
    dynamic_threshold_count = 10
    dynamic_threshold_percent = 1.0
    is_likely_dynamic = False
    if total_unique_ips > 0:
         percent_dynamic = (num_multi_mac_ips / total_unique_ips * 100)
         is_likely_dynamic = (num_multi_mac_ips > dynamic_threshold_count) or \
                             (percent_dynamic > dynamic_threshold_percent)
         print(f"(Dynamic %: {percent_dynamic:.2f}%)")
    elif num_multi_mac_ips > 0:
         is_likely_dynamic = True

    strategy = 'B' if is_likely_dynamic else 'A'

    if strategy == 'B':
        print("🔴 The IP-MAC mapping appears potentially DYNAMIC or involves IP reuse.")
        print("   RECOMMENDATION: Use Strategy B for feature engineering (map IP to Manufacturer/OUI).")
    else:
        print("🟢 The IP-MAC mapping appears predominantly STATIC.")
        print("   RECOMMENDATION: Strategy A for feature engineering (map IP to full MAC) is likely viable.")

    print("\n--- IP-MAC Stability Check Complete ---")
    return strategy, unique_ip_mac_pairs


# --- Main Execution Block ---
if __name__ == "__main__":

    # **** Use the configured paths ****
    pt_file = PREPROCESSED_PT_FILE
    mappings_file = MAPPINGS_JSON_FILE

    # --- Input Validation ---
    if not os.path.isfile(pt_file):
         print(f"❌ Error: Preprocessed data file not found: {pt_file}")
         exit(1)
    if not os.path.isfile(mappings_file):
         print(f"❌ Error: Mappings JSON file not found: {mappings_file}")
         exit(1)

    # --- Run the Check ---
    print(f"Starting stability check using:")
    print(f"  PT File: {pt_file}")
    print(f"  Mappings File: {mappings_file}")

    try:
        recommended_strategy, unique_pairs = check_ip_mac_stability_from_pt(
            pt_file_path=pt_file,
            mappings_json_path=mappings_file,
            ip_col_indices=IP_COL_INDICES,
            mac_col_indices=MAC_COL_INDICES,
            ip_col_names=IP_COL_NAMES,
            mac_col_names=MAC_COL_NAMES
        )

        # Optional: Save the unique pairs DataFrame
        output_csv_path = "unique_ip_mac_pairs_found_from_pt.csv"
        try:
             unique_pairs.to_csv(output_csv_path, index=False)
             print(f"\n💾 Saved unique IP-MAC pairs found to: {output_csv_path}")
        except Exception as e:
             print(f"\n⚠️ Warning: Could not save unique pairs CSV: {e}")

        print(f"\n✅ Final Recommended Strategy: {recommended_strategy}")

    except FileNotFoundError:
         # Should be caught by initial check, but good practice
         print(f"❌ Error: A required file was not found during processing.")
         exit(1)
    except (RuntimeError, ValueError) as e: # Catch specific errors raised in function
         print(f"❌ Error: {e}")
         exit(1)
    except Exception as e:
         print(f"❌ An unexpected error occurred: {type(e).__name__} - {e}")
         exit(1) # Exit with error code

