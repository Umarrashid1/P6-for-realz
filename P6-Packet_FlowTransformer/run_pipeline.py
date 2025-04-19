from pipeline.process import preprocess_flows_as_sequences


DATASET_DIR = '../../dataset/raw_dataset'
OUTPUT_FILE = '../../dataset/packet_small.pt'

preprocess_flows_as_sequences(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    test_mode=True,
    rows_per_file=20000
)
print("Pipeline completed successfully.")


