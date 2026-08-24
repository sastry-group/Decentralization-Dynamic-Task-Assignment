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
    "pdf.fonttype":       42,   
    "ps.fonttype":        42,  
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FIX_PROB = 0.5
FIX_WIN  = 30

BASE_DIR   = "results/logs/4Paper_pareto"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ordered least → most communication
# COMMS_ORDER = ["none", "star", "ring", "brm_12_45", "brm_12", "full"]
COMMS_ORDER = ["none", "rm_12_31_43", "rm_12_31","rm_12","full"]

COMMS_LABELS = {
    "none":      "$\\gamma(G)=5$\n(No-comms)",
    # "none":      "No-comms",
    "brm_12":    "1-edge\nremoved",
    "brm_12_45": "2-edges\nremoved",
    "star":      "Star",
    "ring":      "Ring",
    "full":      "$\\gamma(G)=1$\n(Full-comms)",
    # "full":      "Full-comms",
    "rm_12":       "$\\gamma(G)=2$",
    "rm_12_31":    "$\\gamma(G)=3$",
    "rm_12_31_43": "$\\gamma(G)=4$",
}

# ── Algorithm styles ──────────────────────────────────────────────────────────
ALGO_STYLE = {
    "edd":       ("EDD",       "#4878CF", "o"),
    "hungarian": ("Hungarian", "#D65F5F", "s"),
    "ibr":       ("IBR",       "#6ACC65", "^"),
    "scoba":     ("SCoBA",     "#B47CC7", "D"),
}

def algo_style(algo):
    key = algo.lower()
    if key in ALGO_STYLE:
        return ALGO_STYLE[key]
    return algo.upper(), "#888888", "o"

PROB_MAP = {"05": 0.5, "075": 0.75, "1": 1.0}

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
    r"(?:_darr_(?P<darr>[^_]+))?$"
)

run_folders = sorted(
    [p for p in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(p)]
)

records = []
for folder_path in run_folders:
    run_name  = os.path.basename(folder_path)
    m         = pattern.match(run_name)
    if not m:
        continue
    json_path = os.path.join(folder_path, f"{run_name}.json")
    if not os.path.exists(json_path):
        continue

    meta = m.groupdict()
    raw  = meta.pop("comms_raw").strip("_")
    meta["comms_type"] = re.sub(r'^\d+_', '', raw)
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
    meta["frac_trials"] = frac.tolist()       # per-trial fractions for boxplot
    meta["mean_late"]   = frac.mean()
    meta["time"]        = r.get("avg_time_per_assignment_step_sec", np.nan)
    records.append(meta)

if not records:
    raise ValueError(f"No valid experiment folders found in:\n  {BASE_DIR}")

df = pd.DataFrame(records)
print("comms_type values:", df["comms_type"].unique().tolist())
print("algo values:      ", df["algo"].unique().tolist())

# ── Filter ────────────────────────────────────────────────────────────────────
df_plot = df[
    np.isclose(df["prob"], FIX_PROB) &
    (df["win"] == FIX_WIN) &
    (df["comms_type"].isin(COMMS_ORDER))
].copy()

if df_plot.empty:
    raise ValueError(
        f"No data after filtering. Check FIX_PROB={FIX_PROB}, FIX_WIN={FIX_WIN}."
    )

algo_order  = [a for a in ALGO_STYLE if a in df_plot["algo"].str.lower().unique()]
algorithms  = sorted(df_plot["algo"].unique(),
                     key=lambda a: algo_order.index(a.lower()) if a.lower() in algo_order else 99)
comms_found = [c for c in COMMS_ORDER if c in df_plot["comms_type"].unique()]

print(f"Algorithms  : {algorithms}")
print(f"Comms levels: {comms_found}")

# ── Metadata ──────────────────────────────────────────────────────────────────
ntrials = df["ntrials"].iloc[0]
pkg     = df["pkg"].iloc[0]
dr      = df["dr"].iloc[0]
dep     = df["dep"].iloc[0]
fname_meta = f"dr{dr}_dep{dep}_pkg{pkg}_ntrials{ntrials}"

# ── Layout constants ──────────────────────────────────────────────────────────
n_algos   = len(algorithms)
n_comms   = len(comms_found)
# Each comms group occupies 1 unit on x; algorithms are spread within that
group_w   = 0.7                        # total width allocated per comms group
box_w     = group_w / n_algos * 0.82  # individual box width
x_centers = np.arange(n_comms)        # one tick per comms level

def algo_offsets(n):
    """Return centred offsets for n algorithms within one group."""
    return np.linspace(-group_w / 2 + group_w / (2 * n),
                        group_w / 2 - group_w / (2 * n), n)

offsets = algo_offsets(n_algos)

# ══════════════════════════════════════════════════════════════════════════════
# Figure (a): Boxplot — fraction late
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7.0, 3.8))

for a_idx, algo in enumerate(algorithms):
    name, color, marker = algo_style(algo)
    off = offsets[a_idx]

    box_data = []
    for ct in comms_found:
        rows = df_plot[
            (df_plot["algo"] == algo) &
            (df_plot["comms_type"] == ct)
        ]
        if len(rows) == 0:
            box_data.append(np.array([np.nan]))
        else:
            # concatenate all per-trial arrays
            trials = np.concatenate(rows["frac_trials"].values)
            box_data.append(trials)

    positions = x_centers + off

    bp = ax1.boxplot(
        box_data,
        positions=positions,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        notch=False,
        medianprops=dict(color="white", linewidth=1.8),
        boxprops=dict(facecolor=color, alpha=0.72, linewidth=0.5),
        whiskerprops=dict(color=color, linewidth=0.9),
        capprops=dict(color=color, linewidth=0.9),
    )

    # legend proxy
    ax1.plot([], [], color=color, marker="s", markersize=8,
             linestyle="none", label=name, alpha=0.85)

ax1.set_xticks(x_centers)
ax1.set_xticklabels([COMMS_LABELS.get(c, c) for c in comms_found], fontsize=10)
ax1.set_ylabel("Frac. late packages")
# ax1.set_xlabel("Communication graph structure (undirected)")
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax1.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)
ax1.set_xlim(-0.5, n_comms - 0.5)
ax1.legend(ncol=4, frameon=True, handlelength=1.0,
           columnspacing=0.6, handletextpad=0.3,
           loc="upper right", fontsize=10)
# ax1.text(-0.11, 1.02, "(a)", transform=ax1.transAxes,
#          fontsize=11, fontweight="bold", va="top")

fig1.savefig(os.path.join(OUTPUT_DIR, f"box_late_comms_{fname_meta}.pdf"))
fig1.savefig(os.path.join(OUTPUT_DIR, f"box_late_comms_{fname_meta}.png"))
plt.close(fig1)
print(f"Saved → box_late_comms_{fname_meta}.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure (b): Dot plot — computation time (one dot per algo per comms level)
# time is a single scalar per run so no distribution → use dots + connecting line
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7.0, 2.8))

for a_idx, algo in enumerate(algorithms):
    name, color, marker = algo_style(algo)
    off = offsets[a_idx]

    times = []
    pos   = []
    for c_idx, ct in enumerate(comms_found):
        rows = df_plot[
            (df_plot["algo"] == algo) &
            (df_plot["comms_type"] == ct)
        ]
        if len(rows) == 0:
            continue
        times.append(rows["time"].mean())
        pos.append(x_centers[c_idx] + off)

    times = np.array(times)
    pos   = np.array(pos)

    # connecting line (faint)
    # ax2.plot(pos, times, color=color, linewidth=0.8, alpha=0.4, zorder=2)
    # dots
    ax2.scatter(pos, times, color=color, marker=marker, s=40,
                edgecolors="white", linewidths=0.5, zorder=4, label=name)

ax2.set_yscale("log")
ax2.set_ylabel("Comp. time (s)")
ax2.set_xlabel("Communication graph structure")
ax2.set_xticks(x_centers)
ax2.set_xticklabels([COMMS_LABELS.get(c, c) for c in comms_found], fontsize=10)
ax2.set_xlim(-0.5, n_comms - 0.5)
ax2.grid(axis="y", linewidth=0.5, linestyle=":", color="0.85", zorder=0)
ax2.text(-0.07, 1.02, "(b)", transform=ax2.transAxes,
         fontsize=11, fontweight="bold", va="top")

fig2.savefig(os.path.join(OUTPUT_DIR, f"dot_time_comms_{fname_meta}.pdf"))
fig2.savefig(os.path.join(OUTPUT_DIR, f"dot_time_comms_{fname_meta}.png"))
plt.close(fig2)
print(f"Saved → dot_time_comms_{fname_meta}.pdf")