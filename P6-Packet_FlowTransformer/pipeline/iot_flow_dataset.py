# pipeline/iot_flow_dataset.py
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Dict, Union  # Added Union


class IoTFlowDataset(Dataset):
    """
    Loads a preprocessed .pt file containing flow vectors.
    Expects the .pt file to be a dictionary with keys:
    'numerical_features': Tensor [N_flows, NumFlowNumericalFeatures]
    'categorical_features': Tensor [N_flows, NumFlowCategoricalFeatures] (integer codes)
    'label': Tensor [N_flows]
    """

    def __init__(self, pt_file_path: Union[str, Path]):
        if not isinstance(pt_file_path, (str, Path)):
            raise TypeError(f"pt_file_path must be a string or Path, got {type(pt_file_path)}")

        data_path = Path(pt_file_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"Flow dataset .pt file not found: {data_path}")

        data = torch.load(data_path, map_location="cpu")

        required_keys = ["numerical_features", "categorical_features", "label"]
        for key in required_keys:
            if key not in data:
                raise KeyError(
                    f"Required key '{key}' not found in loaded data from {data_path}. Available keys: {list(data.keys())}")

        self.numerical_features = data["numerical_features"]
        self.categorical_features = data["categorical_features"]
        self.labels = data["label"]

        # --- Sanity checks ---
        num_samples = len(self.labels)
        if not (self.numerical_features.ndim == 2 and self.numerical_features.shape[0] == num_samples):
            raise ValueError(
                f"Numerical features shape mismatch. Expected [N, F_num_flow], got {self.numerical_features.shape} (N={num_samples})."
            )
        # Allow for categorical_features to be empty if no categorical flow features are used
        if self.categorical_features.numel() > 0:  # Check if tensor is not empty
            if not (self.categorical_features.ndim == 2 and self.categorical_features.shape[0] == num_samples):
                raise ValueError(
                    f"Categorical features shape mismatch. Expected [N, F_cat_flow], got {self.categorical_features.shape} (N={num_samples})."
                )
        elif self.categorical_features.ndim == 1 and self.categorical_features.shape[
            0] == 0 and num_samples > 0:  # Empty tensor for categoricals [0]
            # Reshape to [num_samples, 0] if it was saved as a 1D empty tensor from a list
            self.categorical_features = self.categorical_features.reshape(num_samples, 0)
            print(
                f"Note: Categorical features tensor was 1D empty, reshaped to {self.categorical_features.shape} for consistency.")
        elif self.categorical_features.shape[0] != num_samples:  # Catch other inconsistent empty shapes
            raise ValueError(
                f"Categorical features shape mismatch (empty but N not matching). Expected N={num_samples}, got shape {self.categorical_features.shape}."
            )

        if not (self.labels.ndim == 1 and len(self.labels) == num_samples):  # check len() for labels as well
            raise ValueError(
                f"Labels 'label' shape mismatch. Expected [N], but got {self.labels.shape} (N={num_samples})."
            )

        print(
            f"IoTFlowDataset loaded from '{Path(pt_file_path).name}'. "
            f"Numerical shape: {self.numerical_features.shape}, "
            f"Categorical shape: {self.categorical_features.shape}, "
            f"Labels shape: {self.labels.shape}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "numerical_features": self.numerical_features[idx],
            "categorical_features": self.categorical_features[idx],
            "label": self.labels[idx]
        }
        return item