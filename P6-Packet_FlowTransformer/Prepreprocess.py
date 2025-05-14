from utils.category_mapping import generate_and_save_category_mappings
from utils.compute_stats import compute_and_save_global_stats

IS_FLOW = False  # Set to True for flows, False for packets

# Automatically set the dataset directory
DATASET_DIR = "../../../dataset/roni/DatasetFlow" if IS_FLOW else "../../../dataset/raw_dataset"

def main():
    mode = "flow" if IS_FLOW else "packet"
    print(f"▶ Generating categorical mappings ({mode})...")
    generate_and_save_category_mappings(dataset_dir=DATASET_DIR, is_flow=IS_FLOW)

    print(f"▶ Computing global standardization statistics ({mode})...")
    compute_and_save_global_stats(dataset_dir=DATASET_DIR, is_flow=IS_FLOW)

    print(f"✅ Preprocessing complete for {mode} data.")

if __name__ == "__main__":
    main()
