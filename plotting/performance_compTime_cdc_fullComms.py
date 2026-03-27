import re
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import pandas as pd

# ── Publication style ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "errorbar.capsize":   3,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Algorithm display names & colours ────────────────────────────────────────
ALGO_STYLE = {
    # key       display name      colour       marker  linestyle
    "edd":      ("EDD",           "#4878CF",   "o",    "-"),
    "hungarian":("Hungarian",     "#D65F5F",   "s",    "-"),
    "ibr":      ("IBR",           "#6ACC65",   "^",    "-"),
    "scoba":    ("SCoBA",         "#B47CC7",   "D",    "-"),
}
# Fallback for unknown algos
_FALLBACK_COLORS = ["#888888", "#FFAA00", "#00AACC", "#FF6688"]

def algo_style(algo):
    key = algo.lower()
    if key in ALGO_STYLE:
        name, color, marker, ls = ALGO_STYLE[key]
        return name, color, marker, ls
    i = list(set(k for k in [algo])).index(algo) % len(_FALLBACK_COLORS)
    return algo.upper(), _FALLBACK_COLORS[i], "o", "--"

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir   = "results/logs/4Paper_dr15_dp5_steps200_dynT_varyWin"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
run_folders = sorted(
    [p for p in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)]
)
pattern = re.compile(
    r"dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)_ntrials(?P<ntrials>\d+)_nsteps(?P<nsteps>\d+)_"
    r"probpt(?P<prob>\d+)_win(?P<win>\d+)_"
    r"(?P<algo>[^_]+)_"
    r"comms-(?P<comms_radius>\d+)_"
    r"(?P<comms_type>[^_]+)_"
    r"init-(?P<init>[^_]+)_"
    r"dporder-(?P<dporder>.+)"
)

records = []
for folder_path in run_folders:
    run_name = os.path.basename(folder_path)
    m = pattern.match(run_name)
    if not m:
        continue
    json_path = os.path.join(folder_path, f"{run_name}.json")
    if not os.path.exists(json_path):
        continue
    meta = m.groupdict()
    # meta["prob"] = float(meta["prob"]) / 10.0
    PROB_MAP = {"05": 0.5, "075": 0.75, "1": 1.0}
    meta["prob"] = PROB_MAP.get(meta["prob"], float(meta["prob"]) / 10.0)
    with open(json_path) as f:
        r = json.load(f)
    late  = np.array(r["late"],  dtype=float)
    total = np.array(r["total"], dtype=float)
    valid = total > 0
    if not np.any(valid):
        continue
    frac = late[valid] / total[valid]
    meta["mean_late"] = frac.mean()
    meta["sem_late"]  = frac.std(ddof=1) / np.sqrt(len(frac)) if len(frac) > 1 else 0.0
    meta["time"]      = r.get("avg_time_per_assignment_step_sec", np.nan)
    records.append(meta)

if not records:
    raise ValueError("No valid experiment folders found.")

df = pd.DataFrame(records)

# ── Metadata ──────────────────────────────────────────────────────────────────
dr       = df["dr"].iloc[0]
dep      = df["dep"].iloc[0]
pkg      = df["pkg"].iloc[0]
ntrials  = df["ntrials"].iloc[0]
win      = df["win"].iloc[0]

# Filter to full-comms only for this figure
df_full = df[df["comms_type"] == "full"].copy()

algorithms = sorted(df_full["algo"].unique())
probs      = sorted(df_full["prob"].unique())
x          = np.array(probs)

print("Algorithms:", algorithms)
print("Probabilities:", probs)

# ── Helper: collect means & SEMs per algo ─────────────────────────────────────
def collect(metric_mean, metric_sem=None):
    out = {}
    for algo in algorithms:
        means, sems = [], []
        for p in probs:
            rows = df_full[(df_full["algo"] == algo) & (df_full["prob"] == p)]
            if len(rows) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(rows[metric_mean].mean())
                sems.append(rows[metric_sem].mean() if metric_sem else 0.0)
        out[algo] = (np.array(means), np.array(sems))
    return out

late_data = collect("mean_late", "sem_late")
time_data = collect("time")

# ── Figure: 2 rows × 1 col (late + time, same x-axis) ────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(4.5, 5.0),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.08}
)

# ── Row 1: Fraction late ──────────────────────────────────────────────────────
for algo in algorithms:
    name, color, marker, ls = algo_style(algo)
    means, sems = late_data[algo]
    ax1.plot(x, means, color=color, marker=marker, markersize=5,
             linewidth=1.4, linestyle=ls, label=name, zorder=3)
    ax1.fill_between(x, means - sems, means + sems,
                     color=color, alpha=0.15, zorder=2)

ax1.set_ylabel("Mean fraction of late packages")
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax1.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)
ax1.legend(loc="upper left", frameon=True, handlelength=1.8)

# ── Row 2: Computation time (log scale) ───────────────────────────────────────
for algo in algorithms:
    name, color, marker, ls = algo_style(algo)
    means, _ = time_data[algo]
    ax2.plot(x, means, color=color, marker=marker, markersize=5,
             linewidth=1.4, linestyle=ls, label=name, zorder=3)

ax2.set_yscale("log")
ax2.set_ylabel("Avg. time per step (s)")
ax2.set_xlabel("New-request probability")
ax2.set_xticks(probs)
ax2.set_xticklabels([f"{p:.2f}" for p in probs])
ax2.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)

# Shared x-grid
for ax in (ax1, ax2):
    ax.grid(axis="x", linewidth=0.4, linestyle=":", color="0.90", zorder=0)
    ax.set_xlim(min(probs) - 0.05, max(probs) + 0.05)

# Row labels (a) (b)
for ax, lbl in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(-0.13, 1.02, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")

fig_name = (
    f"late_time_prob_"
    f"dr{dr}_dep{dep}_pkg{pkg}_ntrials{ntrials}_win{win}.pdf"
)
plt.savefig(os.path.join(output_dir, fig_name))
plt.savefig(os.path.join(output_dir, fig_name.replace(".pdf", ".png")))
plt.close(fig)
print(f"Saved → {fig_name}")