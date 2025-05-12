import torch
import numpy as np
from pathlib import Path
import pandas as pd
from pipeline.config import categorical_columns, numerical_columns

# === CONFIG ===
OUTPUT_FILE = Path("../../dataset/packet_small.pt")


def validate_pt_file(pt_path: Path):
    assert pt_path.exists(), f"{pt_path} does not exist!"
    data = torch.load(pt_path)
    print(f"\n✅ Loaded: {pt_path}")

    # --- Shape checks ---
    print("\n--- Tensor Shapes ---")
    for key, tensor in data.items():
        print(f"{key:15s}: {tuple(tensor.shape)}")

    # --- NaN / Inf checks ---
    print("\n--- NaN / Inf Checks ---")
    for key, tensor in data.items():
        if torch.is_floating_point(tensor):
            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()
            print(f"{key:15s}: NaNs = {n_nan:<8} Infs = {n_inf}")

    # --- Label distribution ---
    print("\n--- Label Distribution ---")
    labels = data["label"].numpy()
    counts = np.bincount(labels)
    for lbl, cnt in enumerate(counts):
        print(f"Class {lbl}: {cnt} samples")

    # --- Sequence length stats ---
    print("\n--- Sequence Lengths ---")
    attn = data["attention_mask"]
    lengths = attn.sum(dim=1)
    print(f"Min length: {lengths.min().item()}")
    print(f"Max length: {lengths.max().item()}")
    print(f"Avg length: {lengths.float().mean().item():.2f}")

    # --- Numerical feature stats ---
    print("\n--- Standardized Numerical Features ---")
    pkt = data["packet_seq"]
    mask = attn.unsqueeze(-1)
    masked_pkt = pkt * mask
    mean = masked_pkt.sum(dim=(0, 1)) / mask.sum()
    std = (((masked_pkt - mean) ** 2) * mask).sum(dim=(0, 1)).sqrt() / mask.sum().sqrt()
    print(f"Mean (expected ~0): {mean.mean().item():.4f}")
    print(f"Std  (expected ~1): {std.mean().item():.4f}")

    # --- Categorical feature stats ---
    print("\n--- Categorical Feature Ranges ---")
    for col in categorical_columns:
        x = data[col]
        print(f"{col:15s}: min = {x.min().item():<5} max = {x.max().item():<5} shape = {tuple(x.shape)}")

    # --- Length by label ---
    print("\n--- Mean Length per Label ---")
    df = pd.DataFrame({
        "label": labels,
        "length": lengths.numpy()
    })
    print(df.groupby("label")["length"].mean())

    # --- Duplicate sequence check ---
    print("\n--- Duplicate Sequence Check ---")
    hashes = torch.sum(pkt.flatten(1), dim=1)
    n_unique = torch.unique(hashes).numel()
    print(f"Unique sequences: {n_unique} / {len(pkt)}")

# === Run Validation ===
if __name__ == "__main__":
    validate_pt_file(OUTPUT_FILE)
