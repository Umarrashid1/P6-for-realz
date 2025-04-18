from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer
from train.train import train_model, test_model
import pipeline.config as config
from torch.utils.data import random_split
import torch


def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


def load_partial_weights_skip_mlp(model, pretrained_path):
    print(f"\n🔄 Loading weights from: {pretrained_path}")
    pretrained = torch.load(pretrained_path)
    model_dict = model.state_dict()

    matched_weights = {}
    skipped_weights = []
    for k, v in pretrained.items():
        if k.startswith("mlp."):
            skipped_weights.append((k, v.shape, model_dict.get(k, None)))
            continue
        if k in model_dict and v.shape == model_dict[k].shape:
            matched_weights[k] = v
        else:
            skipped_weights.append((k, v.shape, model_dict.get(k, None)))

    model.load_state_dict(matched_weights, strict=False)

    print(f"\n✅ Loaded {len(matched_weights)} layers:")
    for k in matched_weights.keys():
        print(f"   - {k}")

    print(f"\n⚠️ Skipped {len(skipped_weights)} layers:")
    for k, shape_pre, shape_cur in skipped_weights:
        print(f"   - {k}: checkpoint shape {shape_pre} vs. model shape {shape_cur}")


# === Full Dataset ===
dataset_path = "../../dataset/mini_flow.pt"
raw_data = torch.load(dataset_path)
print("Dataset loaded")

# Extract full cardinalities for the whole dataset
categorical_tensor = raw_data["categorical"]
cat_cardinalities = [int(torch.max(categorical_tensor[:, i]) + 1) for i in range(categorical_tensor.shape[1])]

# Init model with 2-layer Transformer
model = IoTTransformer(
    num_numerical=len(config.numerical_columns_flows),
    cat_cardinalities=cat_cardinalities,
    num_classes=8,
    num_layers=2
)

# Load partial weights (skip MLP)
load_partial_weights_skip_mlp(model, "iot_transformer_pretrained_small.pt")

# Load full dataset (no slicing!)
full_dataset = IoTDataset(dataset_path)

# Split into train/val/test
train_dataset, val_dataset, test_dataset = split_dataset_three_ways(full_dataset, val_ratio=0.1, test_ratio=0.1)

print(f"\nTrain size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# Train on full dataset
train_model(model, train_dataset, val_dataset, epochs=5, lr=1e-4)

# Final test
test_model(model, test_dataset)
