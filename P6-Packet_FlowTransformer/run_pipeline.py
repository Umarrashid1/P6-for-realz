from pipeline.process import preprocess_all_in_memory

# Set these to your actual dataset and output path
DATASET_DIR = '../dataset/raw_dataset'
OUTPUT_FILE = '../dataset/packet.pt'

preprocess_all_in_memory(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    test_mode=False,
    rows_per_file=1
)
print("Pipeline completed successfully.")


# This script processes a dataset of CSV files, normalizes numerical columns, and writes the results to a new CSV file.

