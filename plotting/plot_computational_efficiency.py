import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

probs = [0.5]
algos = ["full_ibr_empty_fow", "full_ibr_empty_rand", "full_ibr_empty_reverse", "full_ibr_greedy_rand", "full_ibr_greedy_fow", "full_ibr_greedy_reverse", "full_ibr_random_reverse", "full_ibr_random_fow", "full_ibr_random_random"]
colors = {"full_ibr_empty_fow": "#1f77b4", "full_ibr_empty_rand": "#360eff", "full_ibr_empty_reverse": "#29f8fc", "full_ibr_greedy_rand": "#d62728", "full_ibr_greedy_fow": "#951a0a", "full_ibr_greedy_reverse": "#ef6883", "full_ibr_random_reverse": "#508c4b", "full_ibr_random_fow": "#00ec4f", "full_ibr_random_random": "#4CEE2B"}

base_dir = "results/logs"
# algos = [ "Seq_ibr_greedy_fow","Seq_ibr_greedy_reverse","Seq_ibr_greedy_rand", "Seq_ibr_empty_fow", "Seq_ibr_empty_rand", "Seq_ibr_empty_reverse" ]
# colors = {"Seq_ibr_greedy_fow": "#8c564b", "Seq_ibr_greedy_reverse": "#e377c2", "Seq_ibr_greedy_rand": "#7f7f7f", "Seq_ibr_empty_fow": "#17becf", "Seq_ibr_empty_rand": "#bcbd22", "Seq_ibr_empty_reverse": "#7f7f7f"}

# Extract n_drones / n_depots / window from your sample folder name
sample_folder = f"dr15_dep5_probpt{str(probs[0]).replace('.','')}_win15_{algos[0]}"
m = re.search(r"dr(\d+)_dep(\d+)_probpt(\d+)_win(\d+)", sample_folder)
if m:
    n_drones, n_depots, prob_str, window = m.groups()
else:
    n_drones, n_depots, prob_str, window = "?", "?", "?", "?"

# Hold per-algo time series (averaged per time)
avg_iters = {a: None for a in algos}
avg_k     = {a: None for a in algos}
avg_chg   = {a: None for a in algos}
times     = {a: None for a in algos}

# Load results from CSV and compute per-time averages
for a in algos:
    for p in probs:
        prob_str = str(p).replace('.', '')
        folder = f"dr{n_drones}_dep{n_depots}_probpt{prob_str}_win{window}_{a}"
        folder = os.path.join(base_dir, folder)
        filepath = os.path.join(folder, "computational_efficiency_metrics.csv")  
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        df = pd.read_csv(filepath)
        required_cols = {"trial", "time", "iterations", "k_rounds", "changes"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Warning: {filepath} missing columns: {sorted(missing)}")
            continue

        g = (df.groupby("time", as_index=False)
                .agg(iterations=("iterations", "mean"),
                     k_rounds=("k_rounds", "mean"),
                     changes=("changes", "mean"))
                .sort_values("time"))

        times[a] = g["time"].to_numpy()
        avg_iters[a] = g["iterations"].to_numpy(dtype=float)
        avg_k[a]     = g["k_rounds"].to_numpy(dtype=float)
        avg_chg[a]   = g["changes"].to_numpy(dtype=float)


def plot_metric(metric_dict, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in algos:
        linestyle = "--"
        if times[a] is None or metric_dict[a] is None:
            continue
        if "greedy" in a:
            linestyle = "-"
        ax.plot(times[a], metric_dict[a], marker="o", linestyle=linestyle, linewidth=1, markersize=2,
                label=a, color=colors.get(a))
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"{n_drones} drones, {n_depots} depots, win {window}, prob {probs[0]}")
    fig.tight_layout()
    ax.legend()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {out_name}")

prefix = f"results/k_round_eff_dr{n_drones}_dep{n_depots}_probpt{str(probs[0]).replace('.', '')}_win{window}"
plot_metric(avg_iters, "iterations (avg per time)", f"{prefix}_iterations.png")
plot_metric(avg_k,     "k_rounds (avg per time)",  f"{prefix}_krounds.png")
plot_metric(avg_chg,   "changes (avg per time)",   f"{prefix}_changes.png")
