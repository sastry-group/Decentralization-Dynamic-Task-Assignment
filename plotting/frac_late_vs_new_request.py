import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Configurable params
probs = [0.5]
algos = ["scoba", "edd", "ibr"]
colors = {"scoba": "#1f77b4", "edd": "#ff7f0e", "ibr": "#2ca02c"}  # Optional: custom hex colors
base_dir = "results/logs"

# Initialize result holders
mean_late = {a: [] for a in algos}
sem_late = {a: [] for a in algos}

# Extract n_drones and n_depots from sample file
sample_folder = f"dr5_dep2_probpt{str(probs[0]).replace('.','')}_win10_{algos[0]}"
sample_fn = os.path.join(base_dir, sample_folder, f"{sample_folder}.json")
m = re.search(r"dr(\d+)_dep(\d+)", sample_fn)
n_drones, n_depots = m.groups() if m else ("?", "?")

# Load results
for a in algos:
    for p in probs:
        prob_str = str(p).replace('.', '')
        folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win10_{a}"
        filepath = os.path.join(base_dir, folder, f"{folder}.json")
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        with open(filepath) as f:
            r = json.load(f)
        late = np.array(r["late"]) / np.array(r["total"])
        mean_late[a].append(late.mean())
        sem_late[a].append(
            late.std(ddof=1) / np.sqrt(len(late)) if len(late) > 1 else 0.0
        )

# Plotting
x = np.arange(len(probs))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 4))
for i, a in enumerate(algos):
    ax.bar(
        x + i * width,
        mean_late[a],
        width,
        yerr=sem_late[a],
        capsize=4,
        label=a.upper(),
        color=colors.get(a)
    )

ax.set_xticks(x + (len(algos) - 1) * width / 2)
ax.set_xticklabels([str(p) for p in probs])
ax.set_xlabel("New‐request probability")
ax.set_ylabel("Mean fraction of late packages")
ax.set_ylim(0, 1)
ax.legend()
plt.title(f"{n_drones} drones, {n_depots} depots")

plt.tight_layout()
plt.savefig("results/frac_late_vs_new_request.png")
plt.show()