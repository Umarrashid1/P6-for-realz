from pipeline.process import preprocess_all_in_memory
import pipeline.config as config
# Set these to your actual dataset and output path
DATASET_DIR = '../../dataset/roni'
OUTPUT_FILE = '../../dataset/mini_flow.pt'

categorical_columns = config.categorical_columns_flows
numerical_columns = config.numerical_columns_flows


preprocess_all_in_memory(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    test_mode=True,
    rows_per_file=2000,
    standardize = True
)
print("Pipeline completed successfully.")


# This script processes a dataset of CSV files, normalizes numerical columns, and writes the results to a new CSV file.

