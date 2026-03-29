import re
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.lines as mlines
import glob
import pandas as pd
from collections import defaultdict

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
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Fixed experimental conditions
FIX_PROB = 0.5
FIX_WIN  = 30

BASE_DIR   = "results/logs/4Paper_pareto"   
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Algorithms to include ─────────────────────────────────────────────────────
# Comment out any you don't want. If only one, color is fixed and depot order
# lines are still distinguished by linestyle.
ALGO_INCLUDE = {
    "edd",
    "hungarian",
    "ibr",
    # "scoba",
}

# ── Comms graph ordering: least → most communication ─────────────────────────
# Must match exactly what appears in your folder names as comms_type.
COMMS_ORDER = ["none", "rm_12_31_43", "rm_12_31","rm_12","full"]
COMMS_LABELS = {
    "none":        "Dec. (Tau = 5)",
    "rm_12_31_43": "3 ed. rm (Tau = 4)",
    "rm_12_31":    "2 ed. rm (Tau = 3)",
    "rm_12":       "1 ed. rm (Tau = 2)",
    "full":        "Centralized (Tau = 1)"
}

# ── Depot order styles ────────────────────────────────────────────────────────
# Same color per algorithm, different linestyle per depot order.
DEPOT_ORDER_LIST = ["asc", "desc", "random"]
DEPOT_ORDER_LABELS = {
    "asc":    "Ascending",
    "desc":   "Descending",
    "random": "Random",
}
DEPOT_LINESTYLE = {
    "asc":    "-",
    "desc":   "--",
    "random": ":",
}
DEPOT_MARKER = {
    "asc":    "o",
    "desc":   "s",
    "random": "^",
}

# ── Algorithm colors & markers ────────────────────────────────────────────────
ALGO_STYLE = {
    "edd":       ("EDD",       "#4878CF"),
    "hungarian": ("Hungarian", "#D65F5F"),
    "ibr":       ("IBR",       "#6ACC65"),
    "scoba":     ("SCoBA",     "#B47CC7"),
}
_FALLBACK_COLORS = ["#888888", "#FFAA00", "#00AACC", "#FF6688"]

def algo_color(algo):
    key = algo.lower()
    if key in ALGO_STYLE:
        return ALGO_STYLE[key]
    i = abs(hash(algo)) % len(_FALLBACK_COLORS)
    return algo.upper(), _FALLBACK_COLORS[i]

# ══════════════════════════════════════════════════════════════════════════════

PROB_MAP = {"05": 0.5, "075": 0.75, "1": 1.0}

# ── Regex: captures dporder field ─────────────────────────────────────────────
pattern = re.compile(
    r"dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)_ntrials(?P<ntrials>\d+)_nsteps(?P<nsteps>\d+)_"
    r"probpt(?P<prob>\d+)_win(?P<win>\d+)_"
    r"(?P<algo>[^_]+)_"
    r"comms-(?P<comms_raw>.+?)_"
    r"init-(?P<init>[^_]+)_"
    r"dporder-(?P<dporder>[^_]+)_"
    r"overlap-(?P<overlap>[^_]+)_"
    r"tasks-(?P<tasks>[^_]+)"
    r"(?:_darr_(?P<darr>[^_]+))?"
    r"$"
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
    meta["prob"]  = PROB_MAP.get(meta["prob"], float(meta["prob"]) / 10.0)
    meta["win"]   = int(meta["win"])
    meta["dr"]    = int(meta["dr"])
    meta["dep"]   = int(meta["dep"])
    meta["darr"]  = meta.get("darr") or "nominal"

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
    meta["mean_time"] = r.get("avg_time_per_assignment_step_sec", np.nan)
    records.append(meta)

if not records:
    raise ValueError(f"No valid experiment folders found in:\n  {BASE_DIR}")

df = pd.DataFrame(records)
print("comms_type values:", df["comms_type"].unique().tolist())
print("algo values:      ", df["algo"].unique().tolist())
print("dporder values:   ", df["dporder"].unique().tolist())

# ── Filter ────────────────────────────────────────────────────────────────────
df_plot = df[
    np.isclose(df["prob"], FIX_PROB) &
    (df["win"] == FIX_WIN) &
    (df["algo"].str.lower().isin(ALGO_INCLUDE)) &
    (df["comms_type"].isin(COMMS_ORDER)) &
    (df["dporder"].isin(DEPOT_ORDER_LIST))
].copy()

if df_plot.empty:
    raise ValueError(
        f"No data after filtering.\n"
        f"  FIX_PROB={FIX_PROB}, FIX_WIN={FIX_WIN}\n"
        f"  Available prob={df['prob'].unique()}\n"
        f"  Available win={df['win'].unique()}\n"
        f"  Available algos={df['algo'].unique()}\n"
        f"  Available comms={df['comms_type'].unique()}\n"
        f"  Available dporder={df['dporder'].unique()}"
    )

# preserve algo display order
algo_order = [a for a in ALGO_STYLE if a in df_plot["algo"].str.lower().unique()]
algorithms  = sorted(df_plot["algo"].unique(),
                     key=lambda a: algo_order.index(a.lower()) if a.lower() in algo_order else 99)
comms_found = [c for c in COMMS_ORDER if c in df_plot["comms_type"].unique()]
orders_found = [o for o in DEPOT_ORDER_LIST if o in df_plot["dporder"].unique()]

print(f"Algorithms   : {algorithms}")
print(f"Comms levels : {comms_found}")
print(f"Depot orders : {orders_found}")

# ── x-axis positions ──────────────────────────────────────────────────────────
x      = np.arange(len(comms_found))
xlabels = [COMMS_LABELS.get(c, c) for c in comms_found]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

for algo in algorithms:
    name, color = algo_color(algo)
    for depot in orders_found:
        ls     = DEPOT_LINESTYLE.get(depot, "-")
        marker = DEPOT_MARKER.get(depot, "o")

        means, sems = [], []
        for ct in comms_found:
            rows = df_plot[
                (df_plot["algo"]      == algo) &
                (df_plot["comms_type"] == ct)  &
                (df_plot["dporder"]   == depot)
            ]
            if len(rows) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(rows["mean_late"].mean())
                sems.append(rows["sem_late"].mean())

        means = np.array(means)
        sems  = np.array(sems)

        ax.plot(x, means,
                color=color, linestyle=ls, marker=marker,
                markersize=5, linewidth=1.4, zorder=3)
        ax.fill_between(x, means - sems, means + sems,
                        color=color, alpha=0.08, zorder=2)

# ── Legends ───────────────────────────────────────────────────────────────────
# Legend 1: algorithms (color)
algo_handles = [
    mlines.Line2D([], [], color=algo_color(a)[1], linewidth=1.8,
                  label=algo_color(a)[0])
    for a in algorithms
]

# Legend 2: depot order (linestyle)
depot_handles = [
    mlines.Line2D([], [], color="0.35",
                  linestyle=DEPOT_LINESTYLE.get(d, "-"),
                  marker=DEPOT_MARKER.get(d, "o"),
                  markersize=5, linewidth=1.4,
                  label=DEPOT_ORDER_LABELS.get(d, d))
    for d in orders_found
]
if len(algorithms) > 1:
    # ax.legend(loc="best", frameon=True, handlelength=1.8)
    leg1 = ax.legend(handles=algo_handles,
                    loc="upper left", frameon=True,
                    title="Algorithm", title_fontsize=8,
                    borderpad=0.6, labelspacing=0.35)
    ax.add_artist(leg1)
else:
    ax.legend().set_visible(False)



ax.legend(handles=depot_handles,
          loc="upper right", frameon=True,
          title="Depot order", title_fontsize=8,
          borderpad=0.6, labelspacing=0.35)



# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8.5)
ax.set_xlabel("Communication graph")
ax.set_ylabel("Mean fraction of late packages")
# ax.set_ylim(bottom=0)
# ax.set_ylim(0, 0.5) 
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax.grid(axis="both", linewidth=0.4, linestyle=":", color="0.88", zorder=0)

# x padding
pad = 0.3
ax.set_xlim(min(x) - pad, max(x) + pad)

# panel label
ax.text(-0.10, 1.02, "(a)", transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top")

# ── Save ──────────────────────────────────────────────────────────────────────
dr      = df["dr"].iloc[0]
dep     = df["dep"].iloc[0]
pkg     = df["pkg"].iloc[0]
ntrials = df["ntrials"].iloc[0]

algos_str  = "_".join(sorted(ALGO_INCLUDE))
orders_str = "_".join(sorted(orders_found))

fig_name = (
    f"depotorder_comms_{algos_str}_"
    f"prob{FIX_PROB}_win{FIX_WIN}_"
    f"dr{dr}_dep{dep}_pkg{pkg}_ntrials{ntrials}.pdf"
)
plt.savefig(os.path.join(OUTPUT_DIR, fig_name))
plt.savefig(os.path.join(OUTPUT_DIR, fig_name.replace(".pdf", ".png")))
plt.close(fig)
print(f"Saved → {fig_name}")