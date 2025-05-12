import os
import pandas as pd
import numpy as np
from typing import List, Tuple
# No argparse needed anymore

# --- Import column names from pipeline config ---
try:
    # Assuming the script is run from a location where 'pipeline' is importable
    # Or that pipeline is in the Python path
    from pipeline.config import categorical_columns as packet_categorical_columns
    print("✅ Successfully imported column definitions from pipeline.config")

    # --- Extract relevant IP and MAC columns ---
    # Define expected column names based on the config provided
    EXPECTED_SRC_IP_COL = 'src_ip'
    EXPECTED_DST_IP_COL = 'dst_ip'
    EXPECTED_SRC_MAC_COL = 'src_mac'
    EXPECTED_DST_MAC_COL = 'dst_mac'

    # Verify these columns exist in the imported config list
    if not all(col in packet_categorical_columns for col in [EXPECTED_SRC_IP_COL, EXPECTED_DST_IP_COL, EXPECTED_SRC_MAC_COL, EXPECTED_DST_MAC_COL]):
         missing_in_config = [col for col in [EXPECTED_SRC_IP_COL, EXPECTED_DST_IP_COL, EXPECTED_SRC_MAC_COL, EXPECTED_DST_MAC_COL] if col not in packet_categorical_columns]
         print(f"⚠️ Warning: Expected IP/MAC columns ({missing_in_config}) not found in imported config.categorical_columns. Falling back to hardcoded names.")
         # Fallback to hardcoded names if config import fails or columns are missing
         PACKET_IP_COLS = ['src_ip', 'dst_ip']
         PACKET_MAC_COLS = ['src_mac', 'dst_mac']
    else:
         PACKET_IP_COLS = [EXPECTED_SRC_IP_COL, EXPECTED_DST_IP_COL]
         PACKET_MAC_COLS = [EXPECTED_SRC_MAC_COL, EXPECTED_DST_MAC_COL]
         print(f"   Using IP Columns: {PACKET_IP_COLS}")
         print(f"   Using MAC Columns: {PACKET_MAC_COLS}")

except ImportError:
    print("⚠️ Warning: Could not import from 'pipeline.config'. Ensure the script is run")
    print("   from a location where 'pipeline' is accessible or add it to PYTHONPATH.")
    print("   Falling back to hardcoded default column names: ['src_ip', 'dst_ip'], ['src_mac', 'dst_mac']")
    # Fallback hardcoded names
    PACKET_IP_COLS = ['src_ip', 'dst_ip']
    PACKET_MAC_COLS = ['src_mac', 'dst_mac']
except AttributeError:
     print("⚠️ Warning: 'categorical_columns' not found in imported 'pipeline.config'.")
     print("   Falling back to hardcoded default column names: ['src_ip', 'dst_ip'], ['src_mac', 'dst_mac']")
     # Fallback hardcoded names
     PACKET_IP_COLS = ['src_ip', 'dst_ip']
     PACKET_MAC_COLS = ['src_mac', 'dst_mac']


def check_ip_mac_stability_standalone(
    packet_dataset_dir: str,
    ip_cols: List[str],
    mac_cols: List[str],
    test_mode: bool = False,
    rows_per_file: int = 5000 # Limit rows for faster check in test_mode
    ) -> Tuple[str, pd.DataFrame]:
    """
    Analyzes packet data CSVs in a directory to determine IP-MAC stability.
    Designed to be run standalone, mimicking the style of preprocess_all_in_memory.

    Args:
        packet_dataset_dir: Directory containing packet data CSV files.
        ip_cols: List of IP address column names to check (e.g., ['src_ip', 'dst_ip']).
        mac_cols: List of corresponding MAC address column names (e.g., ['src_mac', 'dst_mac']).
        test_mode: If True, limits rows read per file using rows_per_file.
        rows_per_file: Max rows to read per file when test_mode is True.

    Returns:
        A tuple containing:
          - Recommended strategy ('A' for static, 'B' for dynamic).
          - A DataFrame of unique IP-MAC pairs found.
    """
    print(f"\n--- Starting IP-MAC Stability Check ---")
    print(f"Target Directory: {packet_dataset_dir}")
    print(f"Test Mode: {test_mode} (Rows per file limit: {rows_per_file if test_mode else 'None'})")
    print(f"Checking IP Columns: {ip_cols}")
    print(f"Checking MAC Columns: {mac_cols}")


    all_pairs_list = []
    required_cols = list(set(ip_cols + mac_cols)) # Unique columns needed

    print("\n1. Scanning directory and loading IP-MAC pairs...")
    files_processed = 0
    files_skipped_missing_cols = 0
    files_error_reading = 0

    for root, _, files in os.walk(packet_dataset_dir):
        # Sort files for potentially more consistent processing order (optional)
        files.sort()
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                try:
                    # Check header first efficiently
                    df_header = pd.read_csv(file_path, nrows=0)
                    cols_present_in_header = df_header.columns.tolist()
                    cols_missing_in_header = [col for col in required_cols if col not in cols_present_in_header]

                    if cols_missing_in_header:
                        # Only print warning once per unique set of missing columns if verbose
                        # print(f"[SKIP] Missing columns in {file_path}: {cols_missing_in_header}")
                        files_skipped_missing_cols += 1
                        continue

                    # Load only necessary columns
                    df = pd.read_csv(
                        file_path,
                        usecols=required_cols,
                        nrows=rows_per_file if test_mode else None,
                        low_memory=False # Set low_memory=False for potentially mixed types
                    )

                    # Extract pairs for each IP/MAC column set defined
                    for ip_col, mac_col in zip(ip_cols, mac_cols):
                         # Double check columns exist after loading (should be guaranteed by header check)
                         if ip_col in df.columns and mac_col in df.columns:
                              pairs = df[[ip_col, mac_col]].rename(columns={ip_col: 'IP', mac_col: 'MAC'})
                              all_pairs_list.append(pairs)
                         else:
                              # This indicates a logic error if reached
                              print(f"[ERROR] Column {ip_col} or {mac_col} missing post-load in {file_path}. This shouldn't happen.")

                    files_processed += 1
                    if files_processed % 100 == 0:
                         print(f"   Processed {files_processed} files...")

                except pd.errors.EmptyDataError:
                     # print(f"[SKIP] Empty file: {file_path}") # Can be verbose
                     files_skipped_missing_cols += 1
                except ValueError as ve: # Catch errors often related to usecols
                     print(f"[ERROR] Problem processing columns in {file_path}: {ve}")
                     files_error_reading += 1
                except Exception as e:
                    print(f"[ERROR] Couldn't process {file_path}: {type(e).__name__} - {e}")
                    files_error_reading += 1
                    continue # Skip to next file on error

    print(f"\nFile Processing Summary:")
    print(f"  Files Processed: {files_processed}")
    print(f"  Files Skipped (Missing Cols/Empty): {files_skipped_missing_cols}")
    print(f"  File Reading Errors: {files_error_reading}")

    if not all_pairs_list:
        if files_processed == 0 and files_skipped_missing_cols == 0 and files_error_reading == 0:
             print(f"❌ Error: No CSV files found in directory: {packet_dataset_dir}")
        elif files_processed == 0:
             print(f"❌ Error: No CSV files could be successfully processed (check skips/errors).")
        else:
             print(f"❌ Error: No valid IP-MAC pairs extracted. Check column names ('{ip_cols}', '{mac_cols}') match CSV headers and file contents.")
        # Raise error to stop execution if no data could be gathered
        raise RuntimeError("Failed to extract IP-MAC pairs.")


    print("\n2. Aggregating and cleaning IP-MAC pairs...")
    all_pairs = pd.concat(all_pairs_list, ignore_index=True)
    print(f"   Combined pairs count (raw): {len(all_pairs)}")

    # --- Data Cleaning ---
    # Convert to string first to handle mixed types before cleaning
    all_pairs['IP'] = all_pairs['IP'].astype(str)
    all_pairs['MAC'] = all_pairs['MAC'].astype(str)

    # Define more comprehensive list of potential null/invalid markers
    null_markers = ['nan', 'None', '', '<nil>', 'null', 'NA', 'N/A', '-', '0', '0.0', '00:00:00:00:00:00'] # Add common invalid markers
    all_pairs.replace(null_markers, np.nan, inplace=True)

    # Drop rows where either IP or MAC became NaN after cleaning
    initial_rows = len(all_pairs)
    all_pairs.dropna(subset=['IP', 'MAC'], inplace=True)
    rows_after_nan_drop = len(all_pairs)
    print(f"   Pairs count after dropping NaNs/Invalid markers: {rows_after_nan_drop} (Removed {initial_rows - rows_after_nan_drop})")

    # Drop duplicate IP-MAC pairs
    unique_ip_mac_pairs = all_pairs.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(unique_ip_mac_pairs)
    print(f"   Unique IP-MAC pairs count: {rows_after_dedup} (Removed {rows_after_nan_drop - rows_after_dedup} duplicates)")

    if unique_ip_mac_pairs.empty:
         raise RuntimeError("No valid unique IP-MAC pairs found after cleaning. Check source data quality.")

    print("\n3. Analyzing stability...")
    # Group by IP address and count the number of unique MAC addresses
    ip_stability_counts = unique_ip_mac_pairs.groupby('IP')['MAC'].nunique()

    # Identify IPs associated with more than one MAC address
    multi_mac_ips = ip_stability_counts[ip_stability_counts > 1]

    # 4. Reporting Findings
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
        # Sort by the number of MACs associated (descending) to show most problematic first
        print(multi_mac_ips.sort_values(ascending=False).head(max_examples_to_print))
        if num_multi_mac_ips > max_examples_to_print:
            print(f"... and {num_multi_mac_ips - max_examples_to_print} more.")

        # Provide details for a few problematic IPs
        print("\nExample details for unstable IPs:")
        # Show details for the IPs with the highest number of associated MACs
        for ip_addr in multi_mac_ips.sort_values(ascending=False).head(min(5, num_multi_mac_ips)).index:
             associated_macs = unique_ip_mac_pairs[unique_ip_mac_pairs['IP'] == ip_addr]['MAC'].unique()
             print(f"  - IP: {ip_addr} maps to {len(associated_macs)} MACs: {list(associated_macs)}")
    else:
        print("\nAll analyzed IP addresses map to only one unique MAC address.")

    # 5. Conclusion & Recommendation
    print("\n--- Conclusion ---")
    # Define thresholds (adjust if needed)
    dynamic_threshold_count = 10 # If more than 10 IPs map to multiple MACs
    dynamic_threshold_percent = 1.0 # If more than 1% of unique IPs map to multiple MACs

    is_likely_dynamic = False
    if total_unique_ips > 0: # Avoid division by zero
         percent_dynamic = (num_multi_mac_ips / total_unique_ips * 100)
         is_likely_dynamic = (num_multi_mac_ips > dynamic_threshold_count) or \
                             (percent_dynamic > dynamic_threshold_percent)
         print(f"(Dynamic %: {percent_dynamic:.2f}%)") # Show percentage
    elif num_multi_mac_ips > 0: # Handle case where total_unique_ips is 0 but somehow multi_mac > 0 (shouldn't happen)
         is_likely_dynamic = True


    strategy = 'B' if is_likely_dynamic else 'A'

    if strategy == 'B':
        print("🔴 The IP-MAC mapping appears potentially DYNAMIC or involves IP reuse.")
        print("   RECOMMENDATION: Use Strategy B for feature engineering (map IP to Manufacturer/OUI).")
    else:
        print("🟢 The IP-MAC mapping appears predominantly STATIC.")
        print("   RECOMMENDATION: Strategy A for feature engineering (map IP to full MAC) is likely viable.")

    print("\n--- IP-MAC Stability Check Complete ---")
    # Return strategy and the unique pairs df which might be useful for mapping later
    return strategy, unique_ip_mac_pairs


# --- Main Execution Block ---
if __name__ == "__main__":

    PACKET_DATASET_DIR = '../../dataset/raw_dataset'


    # Set test_mode to True for a faster check on a subset of rows per file
    # Set test_mode to False to analyze all rows (can be slow)
    TEST_MODE_FLAG = True  # Default to True for safety/speed
    ROWS_PER_FILE_LIMIT = 5000 # Used only if TEST_MODE_FLAG is True

    # Use column names derived from config import or fallback
    ip_cols_to_use = PACKET_IP_COLS
    mac_cols_to_use = PACKET_MAC_COLS

    # --- Input Validation ---
    if not os.path.isdir(PACKET_DATASET_DIR):
         print(f"❌ Error: Packet dataset directory not found: {PACKET_DATASET_DIR}")
         exit(1) # Exit with error code
    if len(ip_cols_to_use) != len(mac_cols_to_use):
         print(f"❌ Error: Mismatch between number of IP columns ({len(ip_cols_to_use)}) and MAC columns ({len(mac_cols_to_use)}). Check config or fallback definitions.")
         exit(1)
    if not ip_cols_to_use or not mac_cols_to_use:
          print(f"❌ Error: IP or MAC column lists are empty. Check config import or definitions.")
          exit(1)


    # --- Run the Check ---
    print(f"Starting stability check for hardcoded path: {PACKET_DATASET_DIR}")
    try:
        recommended_strategy, unique_pairs = check_ip_mac_stability_standalone(
            packet_dataset_dir=PACKET_DATASET_DIR,
            ip_cols=ip_cols_to_use,
            mac_cols=mac_cols_to_use,
            test_mode=TEST_MODE_FLAG,
            rows_per_file=ROWS_PER_FILE_LIMIT
        )

        # Optional: Save the unique pairs DataFrame for later use in feature engineering
        output_csv_path = "unique_ip_mac_pairs_found.csv"
        try:
             unique_pairs.to_csv(output_csv_path, index=False)
             print(f"\n💾 Saved unique IP-MAC pairs found to: {output_csv_path}")
        except Exception as e:
             print(f"\n⚠️ Warning: Could not save unique pairs CSV: {e}")

        print(f"\n✅ Final Recommended Strategy: {recommended_strategy}")

    except FileNotFoundError:
         print(f"❌ Error: A file was not found during processing. Check paths and permissions.")
         exit(1)
    except RuntimeError as e:
         print(f"❌ Runtime Error: {e}")
         exit(1)
    except Exception as e:
         print(f"❌ An unexpected error occurred: {type(e).__name__} - {e}")
         exit(1) # Exit with error code

