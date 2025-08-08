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
mean_time = {a: [] for a in algos}
sem_time = {a: [] for a in algos}

# Extract n_drones and n_depots from sample file
sample_folder = f"dr15_dep5_probpt{str(probs[0]).replace('.','')}_win15_{algos[0]}"
sample_fn = os.path.join(base_dir, sample_folder, f"{sample_folder}.json")
m = re.search(r"dr(\d+)_dep(\d+)_probpt(\d+)_win(\d+)", sample_folder)
if m:
    n_drones, n_depots, prob_str, window = m.groups()
else:
    n_drones, n_depots, prob_str, window = "?", "?", "?", "?"

# Load results
for a in algos:
    for p in probs:
        prob_str = str(p).replace('.', '')
        folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win{window}_{a}"
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
        if "avg_time_per_assignment_step_sec" in r:
            avg_time = r["avg_time_per_assignment_step_sec"]
            mean_time[a].append(avg_time)
            sem_time[a].append(0.0)  # Currently no variance across trials
        else:
            print(f"Missing 'avg_time_per_assignment_step_sec' in {filepath}")
            mean_time[a].append(np.nan)
            sem_time[a].append(0.0)

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
plt.savefig(f"results/frac_late_dr{n_drones}_dep{n_depots}_probpt{str(probs[0]).replace('.', '')}_win{window}.png")



# ---- timing plot
fig, ax = plt.subplots(figsize=(6, 4))
for i, a in enumerate(algos):
    ax.bar(
        x + i * width,
        mean_time[a],
        width,
        yerr=sem_time[a],
        capsize=4,
        label=a.upper(),
        color=colors.get(a)
    )

ax.set_xticks(x + (len(algos) - 1) * width / 2)
ax.set_xticklabels([str(p) for p in probs])
ax.set_xlabel("New‐request probability")
ax.set_ylabel("Avg. assignment time per step (s)")
ax.set_title(f"Computation Time — {n_drones} drones, {n_depots} depots")

plt.tight_layout()
ax.legend()
plt.savefig(f"results/time_per_assignment_dr{n_drones}_dep{n_depots}_probpt{str(probs[0]).replace('.', '')}_win{window}.png")