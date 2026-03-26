import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools
import pandas as pd


base_dir = "results/logs/dp5_dr15_pk22_dynT_win30"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Collect run folders
# -----------------------------
run_folders = sorted(
    [p for p in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)]
)

records = []

pattern = re.compile(
    r"dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)_ntrials(?P<ntrials>\d+)_nsteps(?P<nsteps>\d+)_"
    r"probpt(?P<prob>\d+)_win(?P<win>\d+)_"
    r"(?P<algo>[^_]+)_"
    r"comms-(?P<comms_radius>\d+)_"
    r"(?P<comms_type>[^_]+)_"
    r"init-(?P<init>[^_]+)_"
    r"dporder-(?P<dporder>.+)"
)

for folder_path in run_folders:
    run_name = os.path.basename(folder_path)
    m = pattern.match(run_name)
    if not m:
        continue

    json_path = os.path.join(folder_path, f"{run_name}.json")
    if not os.path.exists(json_path):
        continue

    meta = m.groupdict()
    meta["prob"] = float(meta["prob"]) / 10.0

    with open(json_path) as f:
        r = json.load(f)

    late = np.array(r["late"], dtype=float)
    total = np.array(r["total"], dtype=float)

    valid = total > 0
    if not np.any(valid):
        continue

    frac_late = late[valid] / total[valid]

    meta["mean_late"] = frac_late.mean()
    meta["sem_late"] = (
        frac_late.std(ddof=1) / np.sqrt(len(frac_late))
        if len(frac_late) > 1 else 0.0
    )

    meta["time"] = r.get("avg_time_per_assignment_step_sec", np.nan)

    records.append(meta)



# -----------------------------
# Build dataframe
# -----------------------------
if len(records) == 0:
    raise ValueError("No valid experiment folders found.")

df = pd.DataFrame(records)
dr = df["dr"].iloc[0]
dep = df["dep"].iloc[0]
pkg = df["pkg"].iloc[0]
ntrials = df["ntrials"].iloc[0]
win = df["win"].iloc[0]



algorithms = sorted(df["algo"].unique())      # e.g. ['ibr', 'scoba']
comms_levels = sorted(df["comms_type"].unique())  # e.g. ['2', 'full']
probs = sorted(df["prob"].unique())

print("Algorithms found:", algorithms)
print("Communication levels found:", comms_levels)
print("Probabilities found:", probs)


# if len(algorithms) != 2:
#     print("Warning: Script assumes exactly 2 algorithms for clean layout.")

# -----------------------------
# Visual encoding
# -----------------------------

cmap = plt.get_cmap("tab10")
colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
hatch_patterns = ["", "//", "xx", "\\\\", "..", "++"]
hatches = {
    comm: hatch_patterns[i % len(hatch_patterns)]
    for i, comm in enumerate(comms_levels)
}

# colors = {
#     algorithms[0]: "#1f77b4",   # blue
#     algorithms[1]: "#ff7f0e",   # orange
# }

# hatches = {
#     comms_levels[0]: "",
#     # comms_levels[1]: "///"
# }

# bar geometry
n_algos = len(algorithms)
n_comms = len(comms_levels)

group_width = 0.8
bar_width = group_width / (n_algos * n_comms)

x = np.arange(len(probs))

# -----------------------------
# Plot: Fraction Late
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))

for c_idx, comm in enumerate(comms_levels):
    for a_idx, algo in enumerate(algorithms):

        offset = (
            -group_width/2
            + (c_idx * n_algos + a_idx) * bar_width
            + bar_width/2
        )

        means = []
        sems = []

        for p in probs:
            rows = df[
                (df["algo"] == algo) &
                (df["comms_type"] == comm) &
                (df["prob"] == p)
            ]

            if len(rows) == 0:
                means.append(np.nan)
                sems.append(0.0)
            else:
                means.append(rows["mean_late"].mean())
                sems.append(rows["sem_late"].mean())

        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=sems,
            capsize=4,
            color=colors[algo],
            hatch=hatches[comm],
            edgecolor="black",
            label=f"{algo.upper()} – comms {comm}"
        )

# Clean duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(probs)
ax.set_xlabel("New-request probability")
ax.set_ylabel("Mean fraction of late packages")
ax.set_ylim(0, 0.4)

fig_name = (
    f"late_combined_"
    f"dr{dr}_dep{dep}_pkg{pkg}_"
    f"ntrials{ntrials}_"
    f"win{win}_"
    f"prob{probs[0]}.png"
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, fig_name), dpi=200)
plt.close(fig)

# -----------------------------
# Plot: Computation Time
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))

for c_idx, comm in enumerate(comms_levels):
    for a_idx, algo in enumerate(algorithms):

        offset = (
            -group_width/2
            + (c_idx * n_algos + a_idx) * bar_width
            + bar_width/2
        )

        means = []

        for p in probs:
            rows = df[
                (df["algo"] == algo) &
                (df["comms_type"] == comm) &
                (df["prob"] == p)
            ]

            means.append(rows["time"].mean() if len(rows) else np.nan)

        ax.bar(
            x + offset,
            means,
            bar_width,
            color=colors[algo],
            hatch=hatches[comm],
            edgecolor="black",
            label=f"{algo.upper()} – comms {comm}"
        )

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(probs)
ax.set_xlabel("New-request probability")
ax.set_ylabel("Avg. assignment time per step (s)")

fig_name = (
    f"time_combined_"
    f"dr{dr}_dep{dep}_pkg{pkg}_"
    f"ntrials{ntrials}_"
    f"win{win}_"
    f"prob{probs[0]}.png"
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, fig_name), dpi=200)
plt.close(fig)




# # Configurable params
# probs = [0.5]
# algos = ["full_ibr", "none_ibr", "full_scoba", "none_scoba"]
# # algos = ["ibr"]
# # algos = ["ibr", "ibr_TG2","ibr_TG3","ibr_TG4", "ibr_seq", "ibr_un"]
# # algos = ["ibr_random_empty", "ibr_random_greedy", "ibr_reverse_empty", "ibr_reverse_greedy", "ibr_seqOrder_empty", "ibr_seqOrder_greedy"]
# # algos = ["scobaFull", "scobaTG2","scobaTG3","scobaTG4", "scobaSeq", "scobaUn"]
# # algos = ["ibrFull", "ibr"]
# # colors = {"ibr_random_empty": "#1f77b4", "ibr_random_greedy": "#ff7f0e", "ibr_reverse_empty": "#2ca02c", "ibr_reverse_greedy": "#d62728", "ibr_seqOrder_empty": "#9467bd", "ibr_seqOrder_greedy": "#4b828c"}  # Optional: custom hex colors
# # colors = {"ibr": "#1f77b4", "ibrFull": "#ff7f0e"}  # Optional: custom hex colors
# # colors = {"ibr_TG3": "#1f77b4", "ibr": "#ff7f0e", "ibr_TG2": "#2ca02c", "ibr_TG4": "#d62728", "ibr_seq": "#9467bd", "ibr_un": "#4b828c"}  # Optional: custom hex colors
# # colors = {"scobaTG3": "#1f77b4", "scobaFull": "#ff7f0e", "scobaTG2": "#2ca02c", "scobaTG4": "#d62728", "scobaSeq": "#9467bd", "scobaUn": "#4b828c"}  # Optional: custom hex colors

# base_dir = "results/logs"

# # Initialize result holders
# mean_late = {a: [] for a in algos}
# sem_late = {a: [] for a in algos}
# mean_time = {a: [] for a in algos}
# sem_time = {a: [] for a in algos}

# # Extract n_drones and n_depots from sample file
# sample_folder = f"dr15_dep5_probpt{str(probs[0]).replace('.','')}_win15_{algos[0]}"
# sample_fn = os.path.join(base_dir, sample_folder, f"{sample_folder}.json")
# m = re.search(r"dr(\d+)_dep(\d+)_probpt(\d+)_win(\d+)", sample_folder)
# if m:
#     n_drones, n_depots, prob_str, window = m.groups()
# else:
#     n_drones, n_depots, prob_str, window = "?", "?", "?", "?"

# # Load results
# for a in algos:
#     for p in probs:
#         prob_str = str(p).replace('.', '')
#         folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win{window}_{a}"
#         filepath = os.path.join(base_dir, folder, f"{folder}.json")
#         if not os.path.exists(filepath):
#             print(f"Warning: File not found: {filepath}")
#             continue

#         with open(filepath) as f:
#             r = json.load(f)
#         late = np.array(r["late"]) / np.array(r["total"])
#         mean_late[a].append(late.mean())
#         sem_late[a].append(
#             late.std(ddof=1) / np.sqrt(len(late)) if len(late) > 1 else 0.0
#         )
#         if "avg_time_per_assignment_step_sec" in r:
#             avg_time = r["avg_time_per_assignment_step_sec"]
#             mean_time[a].append(avg_time)
#             sem_time[a].append(0.0)  # Currently no variance across trials
#         else:
#             print(f"Missing 'avg_time_per_assignment_step_sec' in {filepath}")
#             mean_time[a].append(np.nan)
#             sem_time[a].append(0.0)

# # Plotting
# x = np.arange(len(probs))
# width = 0.2

# fig, ax = plt.subplots(figsize=(6, 4))

# for i, a in enumerate(algos):
#     ax.bar(
#         x + i * width,
#         mean_late[a],
#         width,
#         yerr=sem_late[a],
#         capsize=4,
#         label=a.upper(),
#         color=colors.get(a)
#     )
#     for j, (m, s) in enumerate(zip(mean_late[a], sem_late[a])):
#         ax.text(
#             x[j] + i * width, 
#             m + s + 0.01,   # above the error bar
#             f"{m:.2f}", 
#             ha="center", va="bottom", fontsize=9
#         )

# ax.set_xticks(x + (len(algos) - 1) * width / 2)
# ax.set_xticklabels([str(p) for p in probs])
# ax.set_xlabel("New‐request probability")
# ax.set_ylabel("Mean fraction of late packages")
# ax.set_ylim(0, 1)
# ax.legend()
# plt.title(f"{n_drones} drones, {n_depots} depots")

# plt.tight_layout()
# plt.savefig(f"results/frac_late_dr{n_drones}_dep{n_depots}_probpt{str(probs[0]).replace('.', '')}_win{window}.png")



# # ---- timing plot
# fig, ax = plt.subplots(figsize=(6, 4))
# for i, a in enumerate(algos):
#     ax.bar(
#         x + i * width,
#         mean_time[a],
#         width,
#         yerr=sem_time[a],
#         capsize=4,
#         label=a.upper(),
#         color=colors.get(a)
#     )

# ax.set_xticks(x + (len(algos) - 1) * width / 2)
# ax.set_xticklabels([str(p) for p in probs])
# ax.set_xlabel("New‐request probability")
# ax.set_ylabel("Avg. assignment time per step (s)")
# ax.set_title(f"Computation Time — {n_drones} drones, {n_depots} depots")

# plt.tight_layout()
# ax.legend()
# plt.savefig(f"results/time_per_assignment_dr{n_drones}_dep{n_depots}_probpt{str(probs[0]).replace('.', '')}_win{window}.png")