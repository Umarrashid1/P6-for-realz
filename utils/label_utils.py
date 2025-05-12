# utils/label_utils.py
import os
from typing import Dict
import logging


# Configure basic logging for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (label_utils) %(message)s')

def find_label_from_path(file_path: str, label_mapping: Dict[str, int]) -> int:
    """
    Determines the label for a file based on its directory path and a provided label mapping.

    It traverses up the directory tree from the file's location. If a directory name
    (case-insensitive) contains any key from the label_mapping, the corresponding
    integer label is returned.

    Args:
        file_path: The absolute path to the file.
        label_mapping: A dictionary mapping label names (strings) to integer labels.
                       Example: {"BenignTraffic": 0, "DDoS": 1}

    Returns:
        The integer label if found, otherwise -1.
    """
    if not label_mapping:
        logging.warning("Label mapping is empty. Cannot determine label from path.")
        return -1

    try:
        current_path = os.path.dirname(os.path.abspath(file_path))
        # Traverse up a limited number of levels to prevent infinite loops on odd paths
        # and to stop at a reasonable project root or dataset root.
        max_levels_up = 10  # Adjust as needed

        for _ in range(max_levels_up):
            if not current_path or current_path == os.path.dirname(current_path):  # Reached root or invalid path
                break

            folder_name = os.path.basename(current_path).lower()  # Case-insensitive comparison

            for key_label_name, label_id_val in label_mapping.items():
                if key_label_name.lower() in folder_name:
                    # logging.debug(f"Found label '{key_label_name}' ({label_id_val}) for path '{file_path}' based on folder '{folder_name}'")
                    return label_id_val

            current_path = os.path.dirname(current_path)  # Move up one level

    except Exception as e:
        logging.error(f"Error finding label for path '{file_path}': {e}")
        return -1  # Return -1 on error

    # logging.warning(f"No label found in path for file: {file_path}")
    return -1  # No label found after checking parent directories


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] (label_utils_test) %(message)s')

    # Define a sample LABEL_MAPPING for testing
    SAMPLE_LABEL_MAPPING = {
        "BenignTraffic": 0,
        "DDoS_Attack": 1,
        "Recon": 2,
        "SpecificDevice_Camera": 3
    }

    # Create dummy directory structure for testing
    test_base_dir = Path("temp_label_test_dir")
    test_base_dir.mkdir(exist_ok=True)

    paths_to_test = {
        test_base_dir / "some_other_folder" / "BenignTraffic_files" / "data1.csv": 0,
        test_base_dir / "DDoS_Attack_logs" / "run1" / "capture.pcap": 1,
        test_base_dir / "archive" / "recon_data_2023" / "scan_results.txt": 2,
        test_base_dir / "iot_devices" / "SpecificDevice_Camera_kitchen" / "stream.csv": 3,
        test_base_dir / "unrelated" / "misc_file.dat": -1,
        test_base_dir / "benigntraffic_mixedcase" / "log.csv": 0,  # Test case-insensitivity
        Path("another_root") / "BenignTraffic" / "file.csv": 0  # Test with a different root
    }

    for d in set(p.parent for p in paths_to_test.keys()):  # Create parent directories
        d.mkdir(parents=True, exist_ok=True)
    for p in paths_to_test.keys():  # Create dummy files
        p.touch()

    print("\n--- Testing find_label_from_path ---")
    for test_path, expected_label in paths_to_test.items():
        # Ensure path is absolute for consistent testing
        abs_test_path = str(test_path.resolve())

        # Create the dummy file if it doesn't exist for the test
        # (Path.touch() might not create parent dirs if they are part of the loop)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.touch(exist_ok=True)

        label_found = find_label_from_path(abs_test_path, SAMPLE_LABEL_MAPPING)
        print(f"Path: {abs_test_path}")
        print(
            f"  Expected: {expected_label}, Found: {label_found} -> {'PASS' if label_found == expected_label else 'FAIL'}")
        assert label_found == expected_label, f"Test failed for {abs_test_path}"

    print("\nTest with empty mapping:")
    label_empty_map = find_label_from_path(str(list(paths_to_test.keys())[0].resolve()), {})
    print(f"  Expected: -1, Found: {label_empty_map} -> {'PASS' if label_empty_map == -1 else 'FAIL'}")
    assert label_empty_map == -1

    # Clean up dummy directory
    import shutil

    # shutil.rmtree(test_base_dir)
    # shutil.rmtree(Path("another_root"), ignore_errors=True)
    print(f"\n(To clean up, manually delete: {test_base_dir} and 'another_root')")

