import re
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import glob
import pandas as pd

# ── Publication style ─────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — change SWEEP to switch between modes
# ══════════════════════════════════════════════════════════════════════════════
SWEEP = "drones"    # "prob"   → x-axis = new-request probability
                    # "window" → x-axis = task window duration
                    # "drones" → x-axis = fleet configuration (dep, dr)

# Fixed values used when the other variables are held constant
FIX_PROB = 0.5     # used when SWEEP = "window" or "drones"
FIX_WIN  = 30       # used when SWEEP = "prob"   or "drones"

# Base directories for each sweep type
BASE_DIRS = {
    "prob":   "results/logs/4Paper_dr15_dp5_steps200_dynT_win30_v2",
    "window": "results/logs/4Paper_dr15_dp5_steps200_dynT_varyWin",
    "drones": "results/logs/4Paper_varyDroneNum",
}
# ══════════════════════════════════════════════════════════════════════════════

assert SWEEP in ("prob", "window", "drones"), \
    "SWEEP must be 'prob', 'window', or 'drones'"

# ── Algorithm styles ──────────────────────────────────────────────────────────
ALGO_STYLE = {
    "edd":       ("EDD",       "#4878CF", "o", "-"),
    "hungarian": ("Hungarian", "#D65F5F", "s", "-"),
    "ibr":       ("IBR",       "#6ACC65", "^", "-"),
    "scoba":     ("SCoBA",     "#B47CC7", "D", "-"),
}
_FALLBACK_COLORS = ["#888888", "#FFAA00", "#00AACC", "#FF6688"]

def algo_style(algo):
    key = algo.lower()
    if key in ALGO_STYLE:
        return ALGO_STYLE[key]
    i = abs(hash(algo)) % len(_FALLBACK_COLORS)
    return algo.upper(), _FALLBACK_COLORS[i], "o", "--"

PROB_MAP = {"05": 0.5, "075": 0.75, "1": 1.0}

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
pattern = re.compile(
    r"dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)_ntrials(?P<ntrials>\d+)_nsteps(?P<nsteps>\d+)_"
    r"probpt(?P<prob>\d+)_win(?P<win>\d+)_"
    r"(?P<algo>[^_]+)_"
    r"comms-(?P<comms_radius>\d+)_"
    r"(?P<comms_type>[^_]+)_"
    r"init-(?P<init>[^_]+)_"
    r"dporder-(?P<dporder>.+)"
)

base_dir     = BASE_DIRS[SWEEP]
run_folders  = sorted(
    [p for p in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)]
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
    meta["prob"] = PROB_MAP.get(meta["prob"], float(meta["prob"]) / 10.0)
    meta["win"]  = int(meta["win"])
    meta["dr"]   = int(meta["dr"])
    meta["dep"]  = int(meta["dep"])

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
    raise ValueError(f"No valid experiment folders found in {base_dir}")

df = pd.DataFrame(records)

# ── Filter to full comms only ─────────────────────────────────────────────────
df_full = df[df["comms_type"] == "full"].copy()

# ── Sweep-specific setup ──────────────────────────────────────────────────────
if SWEEP == "prob":
    df_full      = df_full[df_full["win"] == FIX_WIN].copy()
    sweep_vals   = sorted(df_full["prob"].unique())
    sweep_col    = "prob"
    xlabel       = "New request probability"
    xtick_labels = [f"{v:.2f}" for v in sweep_vals]
    fix_desc     = f"win={FIX_WIN} min"
    fname_tag    = f"prob_fixwin{FIX_WIN}"

elif SWEEP == "window":
    df_full      = df_full[np.isclose(df_full["prob"], FIX_PROB)].copy()
    sweep_vals   = sorted(df_full["win"].unique())
    sweep_col    = "win"
    xlabel       = "Task window duration (min)"
    xtick_labels = [str(v) for v in sweep_vals]
    fix_desc     = f"p={FIX_PROB}"
    fname_tag    = f"win_fixprob{FIX_PROB}"

else:  # "drones"
    df_full = df_full[
        np.isclose(df_full["prob"], FIX_PROB) &
        (df_full["win"] == FIX_WIN)
    ].copy()

    # Build a sortable fleet config key: sort by (dep, dr) ascending
    df_full["fleet"] = list(zip(df_full["dep"], df_full["dr"]))
    fleet_configs    = sorted(df_full["fleet"].unique())   # e.g. [(5,15),(5,50),(10,100)]

    sweep_vals   = list(range(len(fleet_configs)))         # integer positions for x-axis
    sweep_col    = "fleet"
    xlabel       = "Fleet configuration"
    xtick_labels = [f"{dep}dep·{dr}dr" for dep, dr in fleet_configs]
    fix_desc     = f"p={FIX_PROB}, win={FIX_WIN} min"
    fname_tag    = f"drones_fixprob{FIX_PROB}_win{FIX_WIN}"

algorithms = sorted(df_full["algo"].unique())

print(f"Sweep mode  : {SWEEP}")
print(f"Fixed       : {fix_desc}")
print(f"Sweep vals  : {sweep_vals}")
print(f"Algorithms  : {algorithms}")

if len(sweep_vals) == 0:
    raise ValueError(
        f"No data after filtering. Check FIX_PROB={FIX_PROB} and "
        f"FIX_WIN={FIX_WIN} match your folder names."
    )

# ── Collect means & SEMs ──────────────────────────────────────────────────────
def collect(metric_mean, metric_sem=None):
    out = {}
    for algo in algorithms:
        means, sems = [], []

        if SWEEP == "drones":
            for fc in fleet_configs:
                rows = df_full[
                    (df_full["algo"] == algo) &
                    (df_full["fleet"] == fc)
                ]
                if len(rows) == 0:
                    means.append(np.nan)
                    sems.append(np.nan)
                else:
                    means.append(rows[metric_mean].mean())
                    sems.append(rows[metric_sem].mean() if metric_sem else 0.0)
        else:
            for v in sweep_vals:
                rows = df_full[
                    (df_full["algo"] == algo) &
                    (np.isclose(df_full[sweep_col], v))
                ]
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

# ── Metadata for filename ─────────────────────────────────────────────────────
ntrials = df["ntrials"].iloc[0]
pkg     = df["pkg"].iloc[0]

# For drones sweep, dr/dep vary so don't embed them in the filename
if SWEEP == "drones":
    fname_meta = f"pkg{pkg}_ntrials{ntrials}"
else:
    dr  = df["dr"].iloc[0]
    dep = df["dep"].iloc[0]
    fname_meta = f"dr{dr}_dep{dep}_pkg{pkg}_ntrials{ntrials}"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(4.5, 5.0),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.08}
)

x = np.array(sweep_vals)

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
ax1.legend(loc="best", frameon=True, handlelength=1.8)

# ── Row 2: Computation time (log scale) ──────────────────────────────────────
for algo in algorithms:
    name, color, marker, ls = algo_style(algo)
    means, _ = time_data[algo]
    ax2.plot(x, means, color=color, marker=marker, markersize=5,
             linewidth=1.4, linestyle=ls, zorder=3)

ax2.set_yscale("log")
ax2.set_ylabel("Avg. time per step (s)")
ax2.set_xlabel(xlabel)
ax2.set_xticks(x)
ax2.set_xticklabels(xtick_labels)
ax2.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)

# x padding — categorical (drones) gets fixed gap, continuous gets 5% of range
if SWEEP == "drones":
    pad = 0.4
else:
    rng = max(sweep_vals) - min(sweep_vals)
    pad = rng * 0.06 if rng > 0 else 0.05

for ax in (ax1, ax2):
    ax.grid(axis="x", linewidth=0.4, linestyle=":", color="0.90", zorder=0)
    ax.set_xlim(min(x) - pad, max(x) + pad)

# Panel labels
for ax, lbl in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(-0.13, 1.02, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")

# ── Save ──────────────────────────────────────────────────────────────────────
fig_name = f"late_time_{fname_tag}_{fname_meta}.pdf"
plt.savefig(os.path.join(output_dir, fig_name))
plt.savefig(os.path.join(output_dir, fig_name.replace(".pdf", ".png")))
plt.close(fig)
print(f"Saved → {fig_name}")