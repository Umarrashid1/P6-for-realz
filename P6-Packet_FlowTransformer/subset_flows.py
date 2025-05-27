import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import json # Assuming your config.json path is accessible

# Load config to get paths (or hardcode if preferred for a one-off script)
# with open("config.json", 'r') as f:
#     config = json.load(f)
# full_flow_data_path = Path(config['dataset_paths']['processed_flow_pt'])

# Or directly specify the path:
full_flow_data_path = Path("../../../dataset/processed_flows.pt") # Adjust path as needed
output_dir = full_flow_data_path.parent

print(f"Loading full flow data from: {full_flow_data_path}")
full_data = torch.load(full_flow_data_path, map_location='cpu')
labels = full_data['label']
num_samples = len(labels)
indices = list(range(num_samples))

subset_fraction = 0.25 # Example: 10% subset
# Ensure subset_fraction is small enough if some classes have very few samples for stratification
# or handle potential errors from train_test_split if a class has < 2 samples after split.

# Using train_test_split to get a stratified subset of indices
# We only need the 'test' part of the split here, which will be our subset.
# The 'train' part will be the larger remaining part, which we discard for this subset.
_, subset_indices = train_test_split(
    indices,
    test_size=subset_fraction,
    stratify=labels.numpy(),
    random_state=42
)

print(f"Creating a {subset_fraction*100:.0f}% subset with {len(subset_indices)} samples.")

# Create the subset dictionary
subset_data = {
    'numerical_features': full_data['numerical_features'][subset_indices],
    'categorical_features': full_data['categorical_features'][subset_indices],
    'label': full_data['label'][subset_indices],
    'metadata': {
        **(full_data.get('metadata', {})), # Copy original metadata if it exists
        'subset_info': f'Created as a {subset_fraction*100:.0f}% stratified subset from {full_flow_data_path.name}'
    }
}

subset_flow_data_path = output_dir / f"{full_flow_data_path.stem}_subset_{int(subset_fraction*100)}pct.pt"
torch.save(subset_data, subset_flow_data_path)
print(f"Subset saved to {subset_flow_data_path}")