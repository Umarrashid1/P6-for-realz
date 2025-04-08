from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer
from train.train import train_model
from pipeline.config import numerical_columns, categorical_columns
from torch.utils.data import Subset
import torch

# Load full dataset
dataset_path = "../../dataset/packet.pt"
raw_data = torch.load(dataset_path)

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

# Create a subset for quick testing
subset = Subset(full_dataset, list(range(10000)))

# Train
train_model(model, subset, epochs=3)
