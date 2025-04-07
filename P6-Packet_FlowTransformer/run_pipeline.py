from pipeline.process import first_pass, second_pass

# Set these to your actual dataset and output path
DATASET_DIR = '../../dataset/raw_dataset/'
OUTPUT_FILE = '../../dataset/'

first_pass(DATASET_DIR)
second_pass(DATASET_DIR, OUTPUT_FILE)
print("Pipeline completed successfully.")


# This script processes a dataset of CSV files, normalizes numerical columns, and writes the results to a new CSV file.

