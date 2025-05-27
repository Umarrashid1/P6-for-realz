import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import torch
import json
import ast # For literal_eval when loading mappings

# --- Configuration ---
# **** Paths to the preprocessed data and mappings ****
# **** This PT file should be the one generated from PACKET data ****
PREPROCESSED_PT_FILE = '../../dataset/packet_small.pt' # CHANGE THIS if needed
MAPPINGS_JSON_FILE = 'category_mappings.json' # CHANGE THIS if needed

# --- Define the specific column names we need ---
# These MUST match the keys used when saving the .pt file and in the mappings
# (These should ideally come from config, but defining explicitly here for clarity)
SRC_IP_COL_NAME = 'src_ip'
DST_IP_COL_NAME = 'dst_ip'
SRC_MAC_COL_NAME = 'src_mac'
DST_MAC_COL_NAME = 'dst_mac'

IP_COL_NAMES = [SRC_IP_COL_NAME, DST_IP_COL_NAME]
MAC_COL_NAMES = [SRC_MAC_COL_NAME, DST_MAC_COL_NAME]
REQUIRED_KEYS = IP_COL_NAMES + MAC_COL_NAMES


def load_mappings_from_json(path="category_mappings.json") -> Dict[str, Dict[Any, int]]:
    """Loads mappings, converting string representations back to original types."""
    print(f"Loading category mappings from {path}...")

    with open(path, "r") as f:
        raw = json.load(f)
    # Convert repr'd keys back to original types
    loaded_mappings = {}
    for col, mapping in raw.items():
        converted_mapping = {}
        for k_repr, v_int in mapping.items():
            try:
                original_key = ast.literal_eval(k_repr)
                converted_mapping[original_key] = v_int
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"  [Warning] Could not evaluate key '{k_repr}' for column '{col}'. Skipping. Error: {e}")
                continue
        loaded_mappings[col] = converted_mapping
    print("Mappings loaded successfully.")
    return loaded_mappings


def create_reverse_mappings(mappings: Dict[str, Dict[Any, int]]) -> Dict[str, Dict[int, Any]]:
    """Creates reverse mappings (code -> original value)."""
    reverse_mappings = {}
    print("Creating reverse mappings (code -> original value)...")
    for col, mapping in mappings.items():
        reverse_map = {v_int: k_orig for k_orig, v_int in mapping.items()}
        if len(reverse_map) != len(mapping):
             print(f"  [Warning] Potential duplicate codes detected in mapping for column '{col}'.")
        reverse_mappings[col] = reverse_map
    return reverse_mappings


def check_ip_mac_stability_from_pt_v2(
    pt_file_path: str,
    mappings_json_path: str,
    ip_col_names: List[str],
    mac_col_names: List[str]
    ) -> Tuple[str, pd.DataFrame]:
    """
    Analyzes preprocessed data (.pt file with separate categorical keys)
    using category mappings (.json) to determine IP-MAC stability.

    Args:
        pt_file_path: Path to the .pt file containing tensors keyed by column name.
        mappings_json_path: Path to the category_mappings.json file.
        ip_col_names: List of original IP column names (keys in .pt file).
        mac_col_names: List of original MAC column names (keys in .pt file).

    Returns:
        A tuple containing:
          - Recommended strategy ('A' for static, 'B' for dynamic).
          - A DataFrame of unique IP (string) - MAC (string) pairs found.
    """
    print(f"\n--- Starting IP-MAC Stability Check (using .pt file with separate keys) ---")
    print(f"PT File: {pt_file_path}")
    print(f"Mappings File: {mappings_json_path}")

    # 1. Load Mappings and Create Reverse Mappings
    try:
        mappings = load_mappings_from_json(mappings_json_path)
        reverse_mappings = create_reverse_mappings(mappings)
    except Exception as e:
        print(f"❌ Failed to load or process mappings. Cannot proceed.")
        raise

    # 2. Load Preprocessed Data
    print(f"\n1. Loading preprocessed data from {pt_file_path}...")
    try:
        data = torch.load(pt_file_path, map_location='cpu') # Load to CPU
        if not isinstance(data, dict):
             raise TypeError(f".pt file did not contain a dictionary. Found type: {type(data)}")

        # Verify all required keys (column names) exist in the loaded data
        required_keys_in_pt = ip_col_names + mac_col_names
        missing_keys = [key for key in required_keys_in_pt if key not in data]
        if missing_keys:
            raise ValueError(f"Required keys {missing_keys} not found in the .pt file. Available keys: {list(data.keys())}")

        print(f"   Loaded .pt file successfully. Found required keys: {required_keys_in_pt}")
        # Example: Print shape of one tensor
        print(f"   Shape of '{ip_col_names[0]}' tensor: {data[ip_col_names[0]].shape}")

    except FileNotFoundError:
        print(f"❌ Error: Preprocessed data file not found at {pt_file_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")
        raise

    # 3. Reconstruct IP/MAC Strings
    print("\n2. Reconstructing IP and MAC strings from codes...")
    all_pairs_list = []

    # Iterate through the pairs of IP/MAC columns (e.g., src_ip/src_mac, dst_ip/dst_mac)
    for ip_name, mac_name in zip(ip_col_names, mac_col_names):
        print(f"   Processing pair: {ip_name} - {mac_name}")

        # Get the tensors from the loaded data using the names as keys
        ip_codes_tensor = data[ip_name]
        mac_codes_tensor = data[mac_name]

        # Ensure tensors are 1D or 2D (for sequences) - assuming 2D [num_seq, seq_len]
        # Flatten them for mapping if they are sequences
        ip_codes_flat = ip_codes_tensor.flatten().numpy()
        mac_codes_flat = mac_codes_tensor.flatten().numpy()

        # Get the reverse maps for the specific IP and MAC columns
        try:
            reverse_map_ip = reverse_mappings[ip_name]
            reverse_map_mac = reverse_mappings[mac_name]
        except KeyError as e:
             print(f"❌ Error: Column name '{e.args[0]}' not found in loaded mappings. Check config and mapping file consistency.")
             raise

        # Reconstruct IP and MAC strings using the reverse mappings
        ip_strings = [reverse_map_ip.get(code, np.nan) for code in ip_codes_flat]
        mac_strings = [reverse_map_mac.get(code, np.nan) for code in mac_codes_flat]

        pairs = pd.DataFrame({'IP': ip_strings, 'MAC': mac_strings})
        all_pairs_list.append(pairs)

    if not all_pairs_list:
        raise RuntimeError("Failed to reconstruct any IP-MAC pairs.")

    print("\n3. Aggregating and cleaning reconstructed pairs...")
    all_pairs = pd.concat(all_pairs_list, ignore_index=True)
    print(f"   Combined pairs count (raw reconstructed): {len(all_pairs)}")

    # (Cleaning, Deduplication, Analysis, Reporting - Identical to previous script)
    # --- Data Cleaning ---
    initial_rows = len(all_pairs)
    all_pairs.dropna(subset=['IP', 'MAC'], inplace=True)
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
    ip_stability_counts = unique_ip_mac_pairs.groupby('IP')['MAC'].nunique()
    multi_mac_ips = ip_stability_counts[ip_stability_counts > 1]

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

    # Use the configured paths
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
        # Use the specific column names defined at the top
        recommended_strategy, unique_pairs = check_ip_mac_stability_from_pt_v2(
            pt_file_path=pt_file,
            mappings_json_path=mappings_file,
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
         print(f"❌ Error: A required file was not found during processing.")
         exit(1)
    except (RuntimeError, ValueError, TypeError, KeyError) as e: # Catch more specific errors
         print(f"❌ Error: {e}")
         exit(1)
    except Exception as e:
         print(f"❌ An unexpected error occurred: {type(e).__name__} - {e}")
         exit(1)
