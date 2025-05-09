# utils/category_mapping.py
import json
from pathlib import Path

def build_category_mappings(df, categorical_columns):
    mappings = {}
    for col in categorical_columns:
        raw_vals = df[col].dropna().unique().tolist()

        # Ensure consistent string typing for sorting
        str_vals = sorted(str(v) for v in raw_vals)

        # Always include 'unknown' if not present
        if "unknown" not in str_vals:
            str_vals.append("unknown")

        mappings[col] = {val: idx for idx, val in enumerate(str_vals)}
    return mappings

def save_mappings(mappings, path="category_mappings.json"):
    with open(path, "w") as f:
        json.dump(mappings, f)

def load_mappings(path="category_mappings.json"):
    with open(path, "r") as f:
        return json.load(f)
