# utils/category_mapping.py
import json
from pathlib import Path

def build_category_mappings(df, categorical_columns):
    mappings = {}
    for col in categorical_columns:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
    return mappings

def save_mappings(mappings, path="category_mappings.json"):
    with open(path, "w") as f:
        json.dump(mappings, f)

def load_mappings(path="category_mappings.json"):
    with open(path, "r") as f:
        return json.load(f)
