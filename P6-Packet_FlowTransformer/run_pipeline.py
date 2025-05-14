from pipeline.process import create_packet_sequences
import time, pathlib
t0 = time.time()


DATASET_DIR = '../../../dataset/raw_dataset'
OUTPUT_FILE = '../../../dataset/packet_small.pt'

create_packet_sequences(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    test_mode=True,
    rows_per_file=1000,
)

print("Pipeline completed successfully.")
print("elapsed", time.time() - t0)

