from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer
from train.train import train_model, test_model
from pipeline.config import numerical_columns, categorical_columns
from torch.utils.data import Subset, random_split
import torch



def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

# Load full dataset
dataset_path = "../../dataset/packet_small.pt"
raw_data = torch.load(dataset_path)
print("Dataset loaded")

# Extract cardinalities from raw tensor
categorical_tensor = raw_data["categorical"]
cat_cardinalities = [int(torch.max(categorical_tensor[:, i]) + 1) for i in range(categorical_tensor.shape[1])]

# Init model
model = IoTTransformer(
    num_numerical=len(numerical_columns),
    cat_cardinalities=cat_cardinalities,
    num_classes=8
)

# Re-wrap the data using your custom Dataset class
full_dataset = IoTDataset(dataset_path)

# Split the subset into train/val/test
train_dataset, val_dataset, test_dataset = split_dataset_three_ways(full_dataset, val_ratio=0.1, test_ratio=0.1)

# Print subset sizes
print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")


# Train with training and validation sets
train_model(model,
            train_dataset,
            val_dataset,
            epochs=5)

# Test on the held-out test set
test_model(model, test_dataset)





