import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configurable params
probs = [0.5]
# algos = ["ibr"]
# algos = ["ibr", "ibr_TG2","ibr_TG3","ibr_TG4", "ibr_seq", "ibr_un"]
# algos = ["full_ibr_empty_fow", "full_ibr_empty_rand", "full_ibr_empty_reverse", "full_ibr_greedy_rand", "full_ibr_greedy_fow", "full_ibr_greedy_reverse"]
# algos = ["scobaFull", "scobaTG2","scobaTG3","scobaTG4", "scobaSeq", "scobaUn"]
# algos = ["ibrFull", "ibr"]
# colors = {"full_ibr_empty_fow": "#1f77b4", "full_ibr_empty_rand": "#ff7f0e", "full_ibr_empty_reverse": "#2ca02c", "full_ibr_greedy_rand": "#d62728", "full_ibr_greedy_fow": "#9467bd", "full_ibr_greedy_reverse": "#4b828c"}  # Optional: custom hex colors
# colors = {"ibr": "#1f77b4", "ibrFull": "#ff7f0e"}  # Optional: custom hex colors
# colors = {"ibr_TG3": "#1f77b4", "ibr": "#ff7f0e", "ibr_TG2": "#2ca02c", "ibr_TG4": "#d62728", "ibr_seq": "#9467bd", "ibr_un": "#4b828c"}  # Optional: custom hex colors
# colors = {"scobaTG3": "#1f77b4", "scobaFull": "#ff7f0e", "scobaTG2": "#2ca02c", "scobaTG4": "#d62728", "scobaSeq": "#9467bd", "scobaUn": "#4b828c"}  # Optional: custom hex colors
# algos = [ "Seq_ibr_greedy_fow","Seq_ibr_greedy_reverse","Seq_ibr_greedy_rand", "Seq_ibr_empty_fow", "Seq_ibr_empty_rand", "Seq_ibr_empty_reverse" ]
# colors = {"Seq_ibr_greedy_fow": "#8c564b", "Seq_ibr_greedy_reverse": "#e377c2", "Seq_ibr_greedy_rand": "#7f7f7f", "Seq_ibr_empty_fow": "#17becf", "Seq_ibr_empty_rand": "#bcbd22", "Seq_ibr_empty_reverse": "#7f7f7f"}
# algos = ["full_ibr_empty_fow", "full_ibr_empty_rand", "full_ibr_empty_reverse", "full_ibr_greedy_rand", "full_ibr_greedy_fow", "full_ibr_greedy_reverse", "Seq_ibr_greedy_fow","Seq_ibr_greedy_reverse","Seq_ibr_greedy_rand", "Seq_ibr_empty_fow", "Seq_ibr_empty_rand", "Seq_ibr_empty_reverse" ]
# colors = {"full_ibr_empty_fow": "#1f77b4", "full_ibr_empty_rand": "#ff7f0e", "full_ibr_empty_reverse": "#2ca02c", "full_ibr_greedy_rand": "#d62728", "full_ibr_greedy_fow": "#9467bd", "full_ibr_greedy_reverse": "#4b828c", "Seq_ibr_greedy_fow": "#8c564b", "Seq_ibr_greedy_reverse": "#e377c2", "Seq_ibr_greedy_rand": "#7f7f7f", "Seq_ibr_empty_fow": "#17becf", "Seq_ibr_empty_rand": "#bcbd22", "Seq_ibr_empty_reverse": "#7f7f7f"}
algos = ["full_ibr_empty_fow", "full_ibr_empty_rand", "full_ibr_empty_reverse", "full_ibr_greedy_rand", "full_ibr_greedy_fow", "full_ibr_greedy_reverse", "full_ibr_random_reverse", "full_ibr_random_fow", "full_ibr_random_random"]
colors = {"full_ibr_empty_fow": "#1f77b4", "full_ibr_empty_rand": "#360eff", "full_ibr_empty_reverse": "#29f8fc", "full_ibr_greedy_rand": "#d62728", "full_ibr_greedy_fow": "#951a0a", "full_ibr_greedy_reverse": "#ef6883", "full_ibr_random_reverse": "#508c4b", "full_ibr_random_fow": "#00ec4f", "full_ibr_random_random": "#4CEE2B"}


base_dir = "results/logs"
csv_name = "computational_efficiency_metrics.csv"  

sample_folder = f"dr15_dep5_probpt{str(probs[0]).replace('.','')}_win15_{algos[0]}"
m = re.search(r"dr(\d+)_dep(\d+)_probpt(\d+)_win(\d+)", sample_folder)
if m:
    n_drones, n_depots, prob_str, window = m.groups()
else:
    n_drones, n_depots, prob_str, window = "?", "?", "?", "?"

x_sum = {a: [] for a in algos}   # sum of total_krounds across trials
y_mean = {a: [] for a in algos}  # mean late fraction
y_std  = {a: [] for a in algos}  # std  late fraction


# x_trials = {a: [] for a in algos}   # list of per-trial k_rounds
# y_trials = {a: [] for a in algos}   # list of per-trial late fractions

for a in algos:
    for p in probs:
        prob_str = str(p).replace('.', '')
        folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win{window}_{a}"
        folder_path = os.path.join(base_dir, folder)

#         csv_path = os.path.join(folder_path, csv_name)
#         if os.path.exists(csv_path):
#             df = pd.read_csv(csv_path)
#             if {"trial", "time", "k_rounds"}.issubset(df.columns):
#                 per_trial = df.groupby("trial", as_index=False).agg(total_krounds=("k_rounds", "sum"))
#                 x_trials[a].extend(per_trial["total_krounds"].tolist())

#         # --- JSON per-trial late fractions ---
#         json_path = os.path.join(folder_path, f"{folder}.json")
#         if os.path.exists(json_path):
#             with open(json_path, "r") as f:
#                 r = json.load(f)

#             late = np.asarray(r.get("late", []), dtype=float)
#             total = np.asarray(r.get("total", []), dtype=float)

#             mask = (total > 0) & np.isfinite(late)
#             if mask.any():
#                 frac = late[mask] / total[mask]  # per-trial late fractions
#                 y_trials[a].extend(frac.tolist())

# fig, ax = plt.subplots(figsize=(7.5, 5.0))

# for a in algos:
#     xs = np.asarray(x_trials[a], dtype=float)
#     ys = np.asarray(y_trials[a], dtype=float)

#     ax.scatter(
#         xs, ys,
#         s=25,
#         alpha=0.6,
#         color=colors[a],
#         label=a
#     )

# ax.set_xlabel("Total k_round steps")
# ax.set_ylabel("Fraction of late packages")

# ax.grid(True, linestyle="--", alpha=0.5)
# ax.set_title(f"{n_drones} drones, {n_depots} depots, win {window}")
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()



# os.makedirs("results", exist_ok=True)
# out_png = f"results/late_vs_kround_dr{n_drones}_dep{n_depots}_win{window}.png"
# plt.savefig(out_png, dpi=150)
# plt.close(fig)
# print(f"Saved {out_png}")

# os.makedirs("results", exist_ok=True)
# out_png = f"results/late_vs_kround_dr{n_drones}_dep{n_depots}_win{window}.png"
# plt.savefig(out_png, dpi=150)
# plt.close(fig)
# print(f"Saved {out_png}")


        csv_path = os.path.join(folder_path, csv_name) 
        if not os.path.exists(csv_path):
            print(f"Warning: missing {csv_path}")
            x_sum[a].append(np.nan)
        else:
            df = pd.read_csv(csv_path)
            req = {"trial", "time", "k_rounds"}
            if not req.issubset(df.columns):
                print(f"Warning: {csv_path} missing columns: {sorted(req - set(df.columns))}")
                x_sum[a].append(np.nan)
            else:
                # sum k_rounds across time per trial
                per_trial = df.groupby("trial", as_index=False).agg(total_krounds=("k_rounds", "sum"))
                # print(per_trial)
                vals = per_trial["total_krounds"].to_numpy(dtype=int)
                # print(vals)
                x_sum[a].append(float(np.nanmean(vals)) if vals.size else np.nan)

        # --- Y: late fraction per trial from JSON (late/total), then mean/std across trials ---
        json_path = os.path.join(folder_path, f"{folder}.json")
        if not os.path.exists(json_path):
            print(f"Warning: missing {json_path}")
            y_mean[a].append(np.nan); y_std[a].append(0.0)
        else:
            with open(json_path, "r") as f:
                r = json.load(f)
            late = np.asarray(r.get("late", []), dtype=float)
            total = np.asarray(r.get("total", []), dtype=float)
            mask = (total > 0) & np.isfinite(late) & np.isfinite(total)
            if mask.any():
                frac = late[mask] / total[mask]     # per-trial fractions
                y_mean[a].append(float(np.mean(frac)))
                y_std[a].append(float(np.std(frac, ddof=1)) if len(frac) > 1 else 0.0)
            else:
                y_mean[a].append(np.nan); y_std[a].append(0.0)


fig, ax = plt.subplots(figsize=(7.5, 5.0))
for a in algos:
    xs = np.asarray(x_sum[a], dtype=float)
    ys = np.asarray(y_mean[a], dtype=float)
    yerr = np.asarray(y_std[a], dtype=float)


    ax.errorbar(
        xs, ys,
        yerr=yerr, xerr=None,
        marker="o",
        linestyle="-" if len(xs) > 1 else "None",
        linewidth=1, markersize=4,
        color=colors.get(a),
        label=a
    )


ax.set_xlabel("Total k_round steps (sum across steps, mean across trials)")
ax.set_ylabel("Fraction of late packages (mean across trials)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title(f"{n_drones} drones, {n_depots} depots, win {window}")
plt.tight_layout()
ax.legend()

os.makedirs("results", exist_ok=True)
out_png = f"results/late_vs_kround_dr{n_drones}_dep{n_depots}_win{window}.png"
plt.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved {out_png}")