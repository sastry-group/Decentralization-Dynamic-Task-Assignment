import numpy as np
import sys
import os

main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
param_files_folder = os.path.join(main_folder, "param_files")
npz_path = os.path.join(param_files_folder, "scoba_data.npz")
data = np.load(npz_path)

# print("Keys in scoba_data.npz:", list(data.keys()))

# # For each key, show shape and sample values
# for key in data:
#     print(f"\n[{key}] shape = {data[key].shape}")
#     print(data[key][:5]) 



points = data["points"]      # shape: (N, 2)
estimates = data["estimates"]  # shape: (N, N)

print("Points shape:", points.shape)
print("Estimates shape:", estimates.shape)

# Print first 5 GPS points
print("Sample points:\n", points[:5])

# Print estimated travel time between point 0 and 1
print(f"Travel time (0 → 1): {estimates[0,1]:.2f} units")

# Save to CSV for easy inspection
np.savetxt("points.csv", points, delimiter=",", header="lat,lon", comments="")
np.savetxt("travel_time_matrix.csv", estimates, delimiter=",")