import torch
from torch.utils.data import Dataset

class IoTDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.numerical = data["numerical"]
        self.categorical = data["categorical"]
        self.labels = data["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "numerical": self.numerical[idx],
            "categorical": self.categorical[idx],
            "label": self.labels[idx]
        }
