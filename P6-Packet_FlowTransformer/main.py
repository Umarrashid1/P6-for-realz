# main.py
from pipeline.iot_dataset import IoTSequenceDataset
from models.transformer import IoTTransformer
from train.train import train_model, test_model
from torch.utils.data import random_split
import torch


def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


# Load sequence dataset
dataset_path = "../../dataset/dummy.pt"
full_dataset = IoTSequenceDataset(dataset_path, max_seq_len=64)
print("Dataset loaded")

# Init Transformer
sample_input_dim = full_dataset[0]["packet_seq"].shape[1]  # F
model = IoTTransformer(
    input_dim=sample_input_dim,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    num_classes=8,
    max_seq_len=64
)

# Split into train/val/test
train_dataset, val_dataset, test_dataset = split_dataset_three_ways(full_dataset)

print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# Train
train_model(model, train_dataset, val_dataset, epochs=5)

# Test
test_model(model, test_dataset)
