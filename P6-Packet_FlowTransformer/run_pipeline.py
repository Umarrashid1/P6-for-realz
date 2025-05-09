from pipeline.process import preprocess_flows_as_sequences
import time, pathlib
t0 = time.time()


DATASET_DIR = '../../dataset/raw_dataset'
OUTPUT_FILE = '../../dataset/packet.pt'

preprocess_flows_as_sequences(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    test_mode=True,
    rows_per_file=300000
)

print("Pipeline completed successfully.")
print("elapsed", time.time() - t0)

