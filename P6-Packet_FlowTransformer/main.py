from pipeline.iot_dataset import IoTSequenceDataset
from models.transformer import IoTTransformer
from train.train import train_model, test_model
from torch.utils.data import random_split
import torch
from utils.category_mapping import load_mappings
from pipeline.config import categorical_columns_packets, numerical_columns_packets
from pathlib import Path
import json


def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


# Load sequence dataset
dataset_path = "../../../dataset/packet_small.pt"
full_dataset = IoTSequenceDataset(dataset_path, max_seq_len=64)
print("Dataset loaded")

# Init Transformer
cat_map   = load_mappings(is_flow=False)
cat_sizes = {col: len(cat_map[col]) for col in categorical_columns_packets}
cat_pad   = {c: cat_map[c]["unknown"] for c in categorical_columns_packets}


model_arguments = {
    "input_dim": len(numerical_columns_packets),
    "cat_sizes": cat_sizes,
    "cat_padding_idx": cat_pad,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "num_classes": 8, # Num classes for the packet pre-training task
    "max_seq_len": 64,
}

CONFIG_SAVE_DIR = Path("checkpoints_packet_model") # Define a directory for outputs
CONFIG_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Line 1: Ensure directory exists
config_file_path = CONFIG_SAVE_DIR / "packet_config.json" # Line 2: Define full path
with open(config_file_path, 'w') as f: # Line 3
    json.dump(model_arguments, f, indent=4) # Line 4
print(f"✅ Packet model config saved to {config_file_path}") # Line 5 (Status print)

model = IoTTransformer(**model_arguments)

# Split into train/val/test
train_dataset, val_dataset, test_dataset = split_dataset_three_ways(full_dataset)

print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# Train
train_model(model, train_dataset, val_dataset, epochs=5)

# Test
test_model(model, test_dataset)
