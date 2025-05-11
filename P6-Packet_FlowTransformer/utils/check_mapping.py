import json

mapping_path = "category_mappings.json"  # Adjust if needed

with open(mapping_path, "r") as f:
    mappings = json.load(f)

print("=== Category Mapping Summary ===")
for col, mapping in mappings.items():
    num_ids = len(mapping)
    example_keys = list(mapping.keys())[:5]
    print(f"{col:30s} → {num_ids:4d} categories (example keys: {example_keys})")
