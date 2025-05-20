from pipeline.process import create_packet_sequences
import time, pathlib
t0 = time.time()


DATASET_DIR = '../../../dataset/raw_dataset'
OUTPUT_FILE = '../../../dataset/packet_full.pt'

create_packet_sequences(
    dataset_dir=DATASET_DIR,
    output_file=str(OUTPUT_FILE),
    test_mode=False,
    rows_per_file=0,
)

print("Pipeline completed successfully.")
print("elapsed", time.time() - t0)

