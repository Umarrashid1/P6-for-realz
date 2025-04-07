# preprocessing/packet_preprocessor.py

import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
RAW_PACKET_DIR = "data/raw/DI_AD_Packet-based-features"
SAVE_PATH = "data/processed/packet_sequences.pt"
SEQUENCE_LEN = 50

def load_all_csvs(path):
    files = list(Path(path).rglob("*.csv"))
    print(f"Found {len(files)} CSVs.")
    return pd.concat([pd.read_csv(f) for f in tqdm(files)], ignore_index=True)

def preprocess(df):
    print("Original shape:", df.shape)

    # Example: drop obvious non-feature columns
    drop_cols = ['Timestamp', 'Src IP', 'Dst IP', 'Flow ID']  # depends on what you want to keep
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Label encode the target (device type)
    label_col = 'Device'  # <- Confirm this name!
    label_encoder = LabelEncoder()
    df['device_label'] = label_encoder.fit_transform(df[label_col])

    # Drop label from features
    features_df = df.drop(columns=[label_col, 'device_label'])

    # Split numeric/categorical
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = features_df.select_dtypes(include=['object']).columns.tolist()

    # Normalize numeric
    scaler = StandardScaler()
    features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

    # Encode categorical
    for col in cat_cols:
        features_df[col] = LabelEncoder().fit_transform(features_df[col])

    return features_df, df['device_label'], label_encoder, numeric_cols + cat_cols

def group_into_sequences(X, y, sequence_len):
    sequences = []
    targets = []

    # Group by device label
    for device_label in np.unique(y):
        device_data = X[y == device_label]
        num_sequences = len(device_data) // sequence_len

        for i in range(num_sequences):
            seq = device_data[i*sequence_len:(i+1)*sequence_len].values
            sequences.append(seq)
            targets.append(device_label)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

def main():
    df = load_all_csvs(RAW_PACKET_DIR)
    X, y, label_encoder, feature_cols = preprocess(df)
    X_seq, y_seq = group_into_sequences(X, y, SEQUENCE_LEN)

    print(f"Final dataset shape: {X_seq.shape}, labels: {y_seq.shape}")
    torch.save({'X': X_seq, 'y': y_seq}, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
