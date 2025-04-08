import torch
import os
from .process import preprocess_all_in_memory  # <- update import
from .config import numerical_columns, categorical_columns, LABEL_MAPPING

def test_preprocess_subset(tmp_path):
    dataset_dir = '../../../dataset/raw_dataset'
    output_file = '../../../dataset/test_packet.pt'

    preprocess_all_in_memory(
        dataset_dir=dataset_dir,
        output_file=str(output_file),
        test_mode=True,
        rows_per_file=20
    )

    data = torch.load(output_file)

    # Check keys
    assert "numerical" in data
    assert "categorical" in data
    assert "label" in data

    num = data["numerical"]
    cat = data["categorical"]
    label = data["label"]

    # Row count checks
    assert num.shape[0] == cat.shape[0] == label.shape[0], "Row count mismatch between tensors"

    # Column shape checks
    assert num.shape[1] == len(numerical_columns), "Unexpected number of numerical features"
    assert cat.shape[1] == len(categorical_columns), "Unexpected number of categorical features"

    # NaN check
    assert not torch.isnan(num).any(), "Numerical tensor contains NaNs"

    # Label validity check
    valid_labels = set(LABEL_MAPPING.values())
    unique_labels = set(label.tolist())
    assert unique_labels.issubset(valid_labels), f"Invalid labels found: {unique_labels - valid_labels}"

    print(f" Unit test passed — processed {num.shape[0]} rows with labels {sorted(unique_labels)}.")
