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
algos = ["ibr_random_empty", "ibr_random_greedy", "ibr_reverse_empty", "ibr_reverse_greedy", "ibr_seqOrder_empty", "ibr_seqOrder_greedy"]
# algos = ["scobaFull", "scobaTG2","scobaTG3","scobaTG4", "scobaSeq", "scobaUn"]
# algos = ["ibrFull", "ibr"]
colors = {"ibr_random_empty": "#1f77b4", "ibr_random_greedy": "#ff7f0e", "ibr_reverse_empty": "#2ca02c", "ibr_reverse_greedy": "#d62728", "ibr_seqOrder_empty": "#9467bd", "ibr_seqOrder_greedy": "#4b828c"}  # Optional: custom hex colors
# colors = {"ibr": "#1f77b4", "ibrFull": "#ff7f0e"}  # Optional: custom hex colors
# colors = {"ibr_TG3": "#1f77b4", "ibr": "#ff7f0e", "ibr_TG2": "#2ca02c", "ibr_TG4": "#d62728", "ibr_seq": "#9467bd", "ibr_un": "#4b828c"}  # Optional: custom hex colors
# colors = {"scobaTG3": "#1f77b4", "scobaFull": "#ff7f0e", "scobaTG2": "#2ca02c", "scobaTG4": "#d62728", "scobaSeq": "#9467bd", "scobaUn": "#4b828c"}  # Optional: custom hex colors

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

for a in algos:
    for p in probs:
        prob_str = str(p).replace('.', '')
        folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win{window}_{a}"
        folder_path = os.path.join(base_dir, folder)

 
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
                vals = per_trial["total_krounds"].to_numpy(dtype=float)
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


ax.set_xlabel("Total k_round steps (sum across trials)")
ax.set_ylabel("System Efficiency (mean fraction of late packages)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title(f"{n_drones} drones, {n_depots} depots, win {window}")
plt.tight_layout()
ax.legend()

os.makedirs("results", exist_ok=True)
out_png = f"results/late_vs_kround_dr{n_drones}_dep{n_depots}_win{window}.png"
plt.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved {out_png}")