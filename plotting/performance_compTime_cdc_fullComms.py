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
    "font.size":          13,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
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
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SWEEP = "drones"     # "prob"   → x-axis = new-request probability
                    # "window" → x-axis = task window duration
                    # "drones" → x-axis = fleet configuration (dep, dr)
                    # "comms"  → x-axis = communication graph structure
                    # "density"   not all files will have this

# Fixed values used when other variables are held constant
FIX_PROB = 0.5     # used when SWEEP = "window", "drones", or "comms"
FIX_WIN  = 30       # used when SWEEP = "prob", "drones", or "comms"

# For comms sweep: define the display order of your graph structure names.
# Put them in order from least to most communication.
# These must match exactly what appears in your folder names as comms_type.
COMMS_ORDER = ["none", "brm_12", "brm_12_45", "star", "ring", "full"]   

COMMS_LABELS = {
    "none":      "None",
    "brm_12":    "1-edge removed",
    "brm_12_45": "2-edges removed",
    "star":      "Star",
    "ring":      "Ring",
    "full":      "Full",
}

DENSITY_ORDER  = ["broad", "nominal", "narrow"]
DENSITY_LABELS = {
    "broad":   "Low",
    "nominal": "Mid",
    "narrow":  "High",
}

# Base directories
BASE_DIRS = {
    "prob":   "results/logs/4Paper_dr15_dp5_steps200_dynT_win30_v2",
    "window": "results/logs/4Paper_dr15_dp5_steps200_dynT_varyWin",
    "drones": "results/logs/4Paper_varyDroneNum",
    "comms":  "results/logs/4Paper_commsGraphs_multiAlgo",   #
    "density": "results/logs/4Paper_packageDensity"
}


# For comms sweep: algorithms whose compute time is NOT affected by comms
# structure — these are omitted from the time panel to avoid flat clutter.
# COMMS_SKIP_TIME = {"edd", "hungarian"}
COMMS_SKIP_TIME = set()
# ══════════════════════════════════════════════════════════════════════════════

assert SWEEP in ("prob", "window", "drones", "comms", "density"), \
    "SWEEP must be 'prob', 'window', 'drones', 'comms', or 'density'"

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
    r"comms-(?P<comms_raw>.+?)_"
    r"init-(?P<init>[^_]+)_"
    r"dporder-(?P<dporder>[^_]+)_"
    r"overlap-(?P<overlap>[^_]+)_"
    r"tasks-(?P<tasks>[^_]+)"
    r"(?:_darr_(?P<darr>[^_]+))?"   # optional suffix
    r"$"
)



base_dir    = BASE_DIRS[SWEEP]
run_folders = sorted(
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
    raw = meta.pop("comms_raw").strip("_")
    meta["comms_type"] = re.sub(r'^\d+_', '', raw)


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
    raise ValueError(f"No valid experiment folders found in:\n  {base_dir}")

df = pd.DataFrame(records)
# print("comms_type values found:", df["comms_type"].unique().tolist())
# print("algo values found:", df["algo"].unique().tolist())

# ── Sweep-specific filtering & axis setup ─────────────────────────────────────
if SWEEP == "prob":
    df_plot      = df[df["win"] == FIX_WIN].copy()
    sweep_vals   = sorted(df_plot["prob"].unique())
    sweep_col    = "prob"
    xlabel       = "New request probability"
    xtick_labels = [f"{v:.2f}" for v in sweep_vals]
    fix_desc     = f"win={FIX_WIN} min"
    fname_tag    = f"prob_fixwin{FIX_WIN}"
    # full comms only for non-comms sweeps
    df_plot      = df_plot[df_plot["comms_type"] == "full"].copy()

elif SWEEP == "window":
    df_plot      = df[np.isclose(df["prob"], FIX_PROB)].copy()
    df_plot      = df_plot[df_plot["comms_type"] == "full"].copy()
    sweep_vals   = sorted(df_plot["win"].unique())
    sweep_col    = "win"
    xlabel       = "Task window duration (min)"
    xtick_labels = [str(v) for v in sweep_vals]
    fix_desc     = f"p={FIX_PROB}"
    fname_tag    = f"win_fixprob{FIX_PROB}"

elif SWEEP == "drones":
    df_plot = df[
        np.isclose(df["prob"], FIX_PROB) &
        (df["win"] == FIX_WIN) &
        (df["comms_type"] == "full")
    ].copy()
    df_plot["fleet_key"] = (
        df_plot["dep"].astype(str) + "_" + df_plot["dr"].astype(str)
    )
    fleet_configs = sorted(
        df_plot[["dep", "dr"]].drop_duplicates().itertuples(index=False),
        key=lambda row: (row.dep, row.dr)
    )
    sweep_vals   = [f"{r.dep}_{r.dr}" for r in fleet_configs]
    sweep_col    = "fleet_key"
    xlabel       = "Fleet configuration (#depots/#drones)"
    xtick_labels = [f"{r.dep}/{r.dr}" for r in fleet_configs]
    fix_desc     = f"p={FIX_PROB}, win={FIX_WIN} min"
    fname_tag    = f"drones_fixprob{FIX_PROB}_win{FIX_WIN}"

elif SWEEP == "density":
    df_plot = df[
        np.isclose(df["prob"], FIX_PROB) &
        (df["win"] == FIX_WIN) &
        (df["comms_type"] == "full")
    ].copy()
    found_darr = set(df_plot["darr"].unique())
    sweep_vals  = [d for d in DENSITY_ORDER if d in found_darr]
    unlisted    = found_darr - set(DENSITY_ORDER)
    if unlisted:
        print(f"WARNING: darr values in data not in DENSITY_ORDER: {unlisted}")
    sweep_col    = "darr"
    xlabel       = "Spatial conflict level"
    xtick_labels = [DENSITY_LABELS.get(v, v) for v in sweep_vals]
    fix_desc     = f"p={FIX_PROB}, win={FIX_WIN} min"
    fname_tag    = f"density_fixprob{FIX_PROB}_win{FIX_WIN}"


else:  # "comms"
    df_plot = df[
        np.isclose(df["prob"], FIX_PROB) &
        (df["win"] == FIX_WIN)
    ].copy()

    # Use COMMS_ORDER to define x-axis ordering; drop any levels not in data
    found_comms  = set(df_plot["comms_type"].unique())
    sweep_vals   = [c for c in COMMS_ORDER if c in found_comms]

    # Warn about any comms levels in data that weren't listed in COMMS_ORDER
    unlisted = found_comms - set(COMMS_ORDER)
    if unlisted:
        print(f"WARNING: these comms_type values are in your data but not in "
              f"COMMS_ORDER and will be skipped: {unlisted}")
        print(f"  → Add them to COMMS_ORDER in the config section.")

    sweep_col    = "comms_type"
    xlabel       = "Communication graph structure"
    # xtick_labels = sweep_vals          # use the names as-is for tick labels
    xtick_labels = [COMMS_LABELS.get(c, c) for c in sweep_vals]
    fix_desc     = f"p={FIX_PROB}, win={FIX_WIN} min"
    fname_tag    = f"comms_fixprob{FIX_PROB}_win{FIX_WIN}"

algorithms = sorted(df_plot["algo"].unique())

print(f"Sweep mode  : {SWEEP}")
print(f"Fixed       : {fix_desc}")
print(f"Sweep vals  : {sweep_vals}")
print(f"Algorithms  : {algorithms}")

if len(sweep_vals) == 0:
    raise ValueError(
        "No sweep values found after filtering. Check:\n"
        f"  FIX_PROB={FIX_PROB}, FIX_WIN={FIX_WIN}\n"
        f"  COMMS_ORDER={COMMS_ORDER}\n"
        f"  Actual comms_type values in data: {df['comms_type'].unique().tolist()}"
    )

# ── Collect means & SEMs ──────────────────────────────────────────────────────
def collect(metric_mean, metric_sem=None, algo_subset=None):
    algos = algo_subset if algo_subset else algorithms
    out = {}
    for algo in algos:
        means, sems = [], []
        for v in sweep_vals:
            if SWEEP == "drones":
                rows = df_plot[
                    (df_plot["algo"] == algo) &
                    (df_plot[sweep_col] == v)
                ]
            else:
                rows = df_plot[
                    (df_plot["algo"] == algo) &
                    (np.isclose(df_plot[sweep_col].astype(float), float(v))
                     if sweep_col not in ("comms_type", "fleet_key", "darr")
                     else df_plot[sweep_col] == v)
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

# For comms sweep: compute time only for algorithms that actually change
if SWEEP == "comms":
    time_algos = [a for a in algorithms if a.lower() not in COMMS_SKIP_TIME]
else:
    time_algos = algorithms

time_data = collect("time", algo_subset=time_algos)

# ── Metadata for filename ─────────────────────────────────────────────────────
ntrials = df["ntrials"].iloc[0]
pkg     = df["pkg"].iloc[0]

if SWEEP == "drones":
    fname_meta = f"pkg{pkg}_ntrials{ntrials}"
else:
    dr  = df["dr"].iloc[0]
    dep = df["dep"].iloc[0]
    fname_meta = f"dr{dr}_dep{dep}_pkg{pkg}_ntrials{ntrials}"

# ── Figure layout ─────────────────────────────────────────────────────────────
# Comms sweep gets a note under the time panel explaining the omission
comms_note = (
    SWEEP == "comms" and len(COMMS_SKIP_TIME & {a.lower() for a in algorithms}) > 0
)

# fig, (ax1, ax2) = plt.subplots(
#     2, 1,
#     figsize=(3.8, 3.8),
#     sharex=True,
#     gridspec_kw={"height_ratios": [1, 0.75], "hspace": 0.10}
# )
# fig.subplots_adjust(bottom=0.15)
fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))

x = np.arange(len(sweep_vals)) if SWEEP in ("drones", "comms", "density") else np.array(sweep_vals, dtype=float)


# x-axis padding
if SWEEP in ("drones", "comms", "density"):
    pad = 0.4
else:
    rng = float(max(sweep_vals)) - float(min(sweep_vals))
    pad = rng * 0.06 if rng > 0 else 0.05

# for ax in (ax1, ax2):
#     ax.set_xlim(min(x) - pad, max(x) + pad)

# # Panel labels
# for ax, lbl in zip((ax1, ax2), ("(a)", "(b)")):
#     ax.text(-0.18, 1.02, lbl, transform=ax.transAxes,
#             fontsize=10, fontweight="bold", va="top")
# ── Fig (a): Fraction late ──────────────────────────────────────────────────
for algo in algorithms:
    name, color, marker, ls = algo_style(algo)
    means, sems = late_data[algo]
    ax1.plot(x, means, color=color, marker=marker, markersize=5,
             linewidth=1.4, linestyle=ls, label=name, zorder=3)
    ax1.fill_between(x, means - sems, means + sems,
                     color=color, alpha=0.15, zorder=2)

ax1.set_ylabel("Frac. late packages")
ax1.set_xlabel(xlabel)
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax1.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)
ax1.grid(axis="x", linewidth=0.4, linestyle=":", color="0.90", zorder=0)
ax1.set_xticks(x)
ax1.set_xticklabels(xtick_labels, rotation=0 if SWEEP in ("comms", "drones") else 0,
                    ha="right" if SWEEP in ("comms", "drones") else "center")
ax1.set_xlim(min(x) - pad, max(x) + pad)
ax1.legend(ncol=2, frameon=True, handlelength=1.5,
           columnspacing=0.8, handletextpad=0.4, loc="best")
ax1.text(-0.20, 1.02, "(a)", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

fig1.savefig(os.path.join(output_dir, f"late_{fname_tag}_{fname_meta}.pdf"))
fig1.savefig(os.path.join(output_dir, f"late_{fname_tag}_{fname_meta}.png"))
plt.close(fig1)


# ── Figure (b): Computation time  ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))

for algo in time_algos:
    name, color, marker, ls = algo_style(algo)
    means, _ = time_data[algo]
    ax2.plot(x, means, color=color, marker=marker, markersize=5,
             linewidth=1.4, linestyle=ls, label=name, zorder=3)

ax2.set_yscale("log")
ax2.set_ylabel("Comp. time (s)")
ax2.set_xlabel(xlabel)
ax2.set_xticks(x)
ax2.set_xticklabels(xtick_labels, rotation=0 if SWEEP in ("comms", "drones") else 0,
                    ha="right" if SWEEP in ("comms", "drones") else "center")
ax2.set_xlim(min(x) - pad, max(x) + pad)
ax2.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)
ax2.grid(axis="x", linewidth=0.4, linestyle=":", color="0.90", zorder=0)
# ax2.legend(ncol=2, frameon=True, handlelength=1.5,
#            columnspacing=0.8, handletextpad=0.4, loc="best")
ax2.text(-0.20, 1.02, "(b)", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

# Add footnote for comms sweep explaining omitted algorithms
if comms_note:
    skipped = ", ".join(
        ALGO_STYLE[a][0] if a in ALGO_STYLE else a.upper()
        for a in sorted(COMMS_SKIP_TIME & {a.lower() for a in algorithms})
    )
    ax2.annotate(
        f"† {skipped} omitted — compute cost independent of comms structure",
        xy=(0.01, -0.28), xycoords="axes fraction",
        fontsize=7.5, color="0.5", style="italic"
    )



fig2.savefig(os.path.join(output_dir, f"time_{fname_tag}_{fname_meta}.pdf"))
fig2.savefig(os.path.join(output_dir, f"time_{fname_tag}_{fname_meta}.png"))
plt.close(fig2)

print(f"Saved → late_{fname_tag}_{fname_meta}.pdf")
print(f"Saved → time_{fname_tag}_{fname_meta}.pdf")