from pipeline.process import first_pass, second_pass

# Set these to your actual dataset and output path
DATASET_DIR = "/ceph/project/P6-iot-flow-ids/dataset/raw_dataset/DatasetAnomaly/BruteForce"
OUTPUT_FILE = "/ceph/project/P6-iot-flow-ids/dataset/"

first_pass(DATASET_DIR)
second_pass(DATASET_DIR, OUTPUT_FILE)
print("Pipeline completed successfully.")


# This script processes a dataset of CSV files, normalizes numerical columns, and writes the results to a new CSV file.

