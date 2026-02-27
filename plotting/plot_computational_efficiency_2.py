import re
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

base_dir = "results/logs"
csv_name = "computational_efficiency_metrics.csv"

# --- folder name parser (your new format) ---
FOLDER_RE = re.compile(
    r"^dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)__probpt(?P<prob>\d+)_win(?P<win>\d+)"
    r"_(?P<algo>[^_]+)_comms-(?P<commsk>\d+)_(?P<comms>.+?)_init-(?P<init>[^_]+)_dporder-(?P<dporder>[^_]+)$"
)

COLOR_BY = "comms" # options: "comms", "init", or None 

# Color by comms (optional)
COMMS_COLORS = {
    "full": "#1f77b4",
    "ring": "#ff7f0e",
    "none": "#2ca02c",
    "edge_rm_12_T2": "#d62728",
    "edge_rm_12_31_T3": "#9467bd",
    "edge_rm_12_31_43_T4": "#8c564b",
}

INIT_COLORS = {
    "greedy": "#1f77b4",
    "random": "#ff7f0e",
    "empty":  "#2ca02c",
}

INIT_SHADE = {
    "greedy": 1.0,
    "random": 0.7,
    "empty": 0.4,
}

DPORDER_MARKER = {
    "asc": "o",
    "desc": "s",
    "random": "^",
}


DPORDER_SHADE = {"asc": 1.00, "desc": 0.75, "random": 0.55}

# Hold per-run time series
times = {}
series = {}     # series[(folder, metric_name)] = np array
labels = {}
colors = {}

# Metrics you want as TIME SERIES plots (must exist as CSV columns)
METRICS = [
    ("iterations", "iterations (avg per time)"),
    ("k_rounds", "k_rounds (avg per time)"),
    ("changes", "changes (avg per time)"),

    # extras you added
    ("agent_updates", "agent_updates (avg per time)"),
    ("candidate_evals", "candidate_evals (avg per time)"),
    ("sum_eval_work", "sum_eval_work (avg per time)"),
    ("avg_visible_degree", "avg_visible_degree (avg per time)"),
    ("avg_eval_work", "avg_eval_work (avg per time)"),
    ("avg_competitors_per_eval", "avg_competitors_per_eval (avg per time)"),
]


def shade_color(color, factor):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))

def shade(color, k=0, n=1, factor=0.25):
    """
    If n == 1 → return base color.
    If n > 1 → return k-th lighter/darker shade.
    """
    if n <= 1:
        return color
    rgb = np.array(mcolors.to_rgb(color))
    # spread shades around base color
    delta = (k - (n - 1) / 2) * factor
    return tuple(np.clip(rgb + delta, 0, 1))


def parse_folder_name(folder: str):
    m = FOLDER_RE.match(folder)
    return m.groupdict() if m else None

def label_from_meta(meta: dict):
    # legend based on comms if full/none/ring, init if greedy, drone order ascending
    return f"{meta['comms']} | init={meta['init']} | order={meta['dporder']}"

def find_matching_folders(base_dir: str, require: dict):
    """
    require: dict of fixed fields to match (strings), any None means "don't care"
    """
    out = []
    for folder in os.listdir(base_dir):
        meta = parse_folder_name(folder)
        if not meta:
            continue
        ok = True
        for k, v in require.items():
            if v is None:
                continue
            if str(meta.get(k)) != str(v):
                ok = False
                break
        if ok:
            out.append((folder, meta))
    out.sort(key=lambda x: (x[1]["comms"], x[1]["init"], x[1]["dporder"], x[0]))
    return out

# ====== YOU CONTROL WHAT GETS READ HERE ======
# Example: read all runs with dr=15, dep=5, pkgnum=22, prob=0.5, win=15, algo=ibr, commsk=5
# and let comms vary among full/ring/none (or any others you have)
require = {
    "dr": "100",
    "dep": "10",
    "pkg": "150",
    "prob": "05",
    "win": "30",
    "algo": "ibr",
    "commsk": "10",
    # comms/init/dporder left None => plot all that match
    "comms": None,
    "init": "greedy",
    "dporder": "asc",
}

runs = find_matching_folders(base_dir, require)
if not runs:
    raise RuntimeError("No matching folders found. Check base_dir and require filters.")





# Load CSVs and compute per-time averages
for folder, meta in runs:
    folder_path = os.path.join(base_dir, folder)
    filepath = os.path.join(folder_path, csv_name)
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        continue

    df = pd.read_csv(filepath)
    if not {"trial", "time"}.issubset(df.columns):
        print(f"Warning: {filepath} missing trial/time columns")
        continue

    # per-time mean across trials
    g = df.groupby("time", as_index=False).mean(numeric_only=True).sort_values("time")
    times[folder] = g["time"].to_numpy(dtype=float)

    labels[folder] = label_from_meta(meta)
    if COLOR_BY == "comms":
        colors[folder] = COMMS_COLORS.get(f"{meta['commsk']}_{meta['comms']}", COMMS_COLORS.get(meta["comms"], "#000000"))

    elif COLOR_BY == "init":
        base = INIT_COLORS.get(meta["init"], "#000000")
        colors[folder] = shade_color(base, DPORDER_SHADE.get(meta["dporder"], 1.0))
    else:
        base = "#000000"
    # colors[folder] = COMMS_COLORS.get(meta["comms"], None)
    # base = COMMS_COLORS.get(meta["comms"], "#000000")
    # base = COMMS_COLORS.get(meta["comms"], "#000000")
    # shade = INIT_SHADE.get(meta["init"], 1.0)
    # colors[folder] = base
    # colors[folder] = shade_color(base, INIT_SHADE.get(meta["init"], 1.0))

    # store each metric series if present
    for col, _ylabel in METRICS:
        if col in g.columns:
            series[(folder, col)] = g[col].to_numpy(dtype=float)
        else:
            series[(folder, col)] = None  # keep key so plotting loops are simple


def plot_metric(metric_col: str, ylabel: str, out_name: str, title_meta: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    for folder, meta in runs:
        if times.get(folder) is None:
            continue
        ys = series.get((folder, metric_col))
        if ys is None:
            continue

        linestyle = "-" if meta.get("init") == "greedy" else "--"
        ax.plot(
            times[folder], ys,
            marker=DPORDER_MARKER.get(meta["dporder"], "o"),
            linestyle=linestyle,
            linewidth=1,
            markersize=2,
            label=labels[folder],
            color=colors.get(folder)
        )

    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{title_meta['dr']} drones, {title_meta['dep']} depots, pkg {title_meta['pkg']}, "
        f"win {title_meta['win']}, probpt {title_meta['prob']}"
    )
    fig.tight_layout()
    ax.legend()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {out_name}")

# Title fields from first run
m0 = runs[0][1]
prefix = f"results/time_series_dr{m0['dr']}_dep{m0['dep']}_pkg{m0['pkg']}_probpt{m0['prob']}_win{m0['win']}"

for col, ylabel in METRICS:
    plot_metric(col, ylabel, f"{prefix}_{col}.png", m0)

