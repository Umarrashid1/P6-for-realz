import numpy as np
stats = np.load("standardization_stats_flows.npz") # Make sure this is the correct file
print("Means:", stats['mean'])
print("Std Devs:", stats['std'])
print("Medians:", stats['median'])
# Check for NaNs or Infs in these arrays
if np.isnan(stats['mean']).any() or np.isinf(stats['mean']).any():
    print("WARNING: NaNs/Infs in loaded means!")
if np.isnan(stats['std']).any() or np.isinf(stats['std']).any():
    print("WARNING: NaNs/Infs in loaded stds!")