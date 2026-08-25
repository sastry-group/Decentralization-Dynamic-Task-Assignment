"""
Fig 4: Empirical IBR efficiency across communication graphs,
       with multiple lines for different parameter settings.

Each line = one (prob, win, fleet) configuration.
x-axis  = τ(G)
y-axis  = Eff_emp = completion_rate(G) / completion_rate(full)

VUG bounds can be toggled on/off via SHOW_VUG_BOUNDS.
"""

import re, json, os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.lines as mlines
import pandas as pd

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
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
BASE_DIR   = "results/logs/4Paper_PoA2"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ALGO = "ibr"
DEPOT_ORDER = "random"       # or None to average over all
FULL_COMMS_KEY = "full"

# ── Toggle VUG bound overlays ─────────────────────────────────────────────────
SHOW_VUG_BOUNDS = False

# ── Define the parameter configurations to plot ──────────────────────────────
# Each entry: (label, color, marker, linestyle, filter_dict)
# filter_dict keys: prob, win, dr, dep, darr — only include keys you want to fix
# The script will find all matching runs and group by comms_type.

SWEEP_CONFIGS = [
    # --- Probability sweep (fix win=30, 5dep/15dr) ---
    # ("$p=0.5,\\; w=30$",   "#2166AC", "o", "-",  {"prob": 0.5,  "win": 30, "dr": 15, "dep": 5}),
    # ("$p=0.75,\\; w=30$",  "#67A9CF", "o", "--",  {"prob": 0.75, "win": 30, "dr": 15, "dep": 5}),
    ("$p=1.0,\\; w=30$",   "#2286BD", "o", "--",  {"prob": 1.0,  "win": 30, "dr": 15, "dep": 5}),

    # --- Window sweep (fix prob=0.5, 5dep/15dr) ---
    # ("$p=0.5,\\; w=15$",   "#7A18B2", "D", "--", {"prob": 0.5, "win": 15, "dr": 15, "dep": 5}),
    ("$p=0.5,\\; w=45$",   "#D67EEC", "D", "--", {"prob": 0.5, "win": 45, "dr": 15, "dep": 5}),

    # --- Hardest setting ---
    # ("$p=1.0,\\; w=15$",   "#B2182B", "D", "--", {"prob": 1.0, "win": 15, "dr": 15, "dep": 5}),

    # --- Fleet configs (fix prob=0.5, win=30) ---
    ("5 depots / 15 drones",  "#18B225", "v", "--",  {"prob": 0.5, "win": 30, "dr": 15, "dep": 5}),
    # ("5 dep / 50 dr",  "#1B7837", "v", "--",  {"prob": 0.5, "win": 30, "dr": 50, "dep": 5}),
    # ("6 dep / 60 dr","#A6DBA0", "P", ":",  {"prob": 0.5, "win": 30, "dr": 60, "dep": 6}),

    # --- Spatial conflict (if you have darr field) ---
    # ("Low conflict",   "#F04029", ">", "--", {"prob": 0.5, "win": 45, "dr": 15, "dep": 5, "darr": "broad"}),
    ("High conflict",  "#CA0020", ">", "--", {"prob": 0.5, "win": 45, "dr": 15, "dep": 5, "darr": "narrow"}),
]

# ══════════════════════════════════════════════════════════════════════════════
PROB_MAP = {"05": 0.5, "075": 0.75, "1": 1.0}

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

# ── Load all data ─────────────────────────────────────────────────────────────
run_folders = sorted(
    [p for p in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(p)]
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
    raw = meta.pop("comms_raw").strip("_")
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

    frac_late = late[valid] / total[valid]
    frac_completed = 1.0 - frac_late

    meta["mean_completed"] = frac_completed.mean()
    meta["sem_completed"]  = frac_completed.std(ddof=1) / np.sqrt(len(frac_completed)) if len(frac_completed) > 1 else 0.0

    cm = r.get("comms_metrics", {})
    meta["tau"]               = cm.get("tau", None)
    meta["alpha_star"]        = cm.get("alpha_star_reciprocal", None)
    meta["alpha"]             = cm.get("alpha_reciprocal", None)
    meta["poa_lb_general"]    = cm.get("poa_lb_general", None)
    meta["poa_lb_consistent"] = cm.get("poa_lb_consistent", None)
    meta["poa_ub"]            = cm.get("poa_ub", None)

    records.append(meta)

if not records:
    raise ValueError(f"No valid experiment folders found in:\n  {BASE_DIR}")

df = pd.DataFrame(records)

# ── Filter to target algo + depot order ───────────────────────────────────────
mask = df["algo"].str.lower() == TARGET_ALGO
if DEPOT_ORDER is not None:
    mask = mask & (df["dporder"] == DEPOT_ORDER)
df_algo = df[mask].copy()

print(f"Total {TARGET_ALGO} runs loaded: {len(df_algo)}")
print(f"Comms types: {sorted(df_algo['comms_type'].unique())}")
print(f"Prob values: {sorted(df_algo['prob'].unique())}")
print(f"Win values:  {sorted(df_algo['win'].unique())}")
print(f"Fleet:       {sorted(df_algo[['dep','dr']].drop_duplicates().values.tolist())}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 2.0))

any_plotted = False

for label, color, marker, ls, filters in SWEEP_CONFIGS:
    # Apply filters
    sub = df_algo.copy()
    for key, val in filters.items():
        if key == "prob":
            sub = sub[np.isclose(sub["prob"], val)]
        elif key in sub.columns:
            sub = sub[sub[key] == val]

    if sub.empty:
        print(f"  WARNING: no data for '{label}' — skipping")
        continue

    # Get full-comm baseline for this parameter setting
    full_rows = sub[sub["comms_type"] == FULL_COMMS_KEY]
    if full_rows.empty:
        print(f"  WARNING: no full-comm baseline for '{label}' — skipping")
        continue
    full_completion = full_rows["mean_completed"].mean()

    # Aggregate by comms_type
    agg = sub.groupby("comms_type").agg(
        mean_completed=("mean_completed", "mean"),
        sem_completed=("sem_completed", "mean"),
        tau=("tau", "first"),
        alpha_star=("alpha_star", "first"),
        poa_lb_consistent=("poa_lb_consistent", "first"),
    ).reset_index()

    agg["eff_emp"]     = agg["mean_completed"] / full_completion
    agg["eff_emp_sem"] = agg["sem_completed"] / full_completion
    agg = agg.sort_values("tau").reset_index(drop=True)

    tau_vals = agg["tau"].values
    eff_vals = agg["eff_emp"].values
    sem_vals = agg["eff_emp_sem"].values

    ax.errorbar(tau_vals, eff_vals, yerr=sem_vals,
                color=color, marker=marker, markersize=6,
                linewidth=1., linestyle=ls, capsize=3,
                label=label, zorder=4)
    # ax.errorbar(tau_vals, eff_vals, yerr=sem_vals,
    #             color=color, marker=marker, markersize=3,
    #             linestyle="none",
    #             capsize=3, label=label, zorder=4, alpha=0.7)
    any_plotted = True

    # Print summary
    print(f"\n  {label}  (full completion = {full_completion:.4f})")
    for _, row in agg.iterrows():
        print(f"    τ={row['tau']:.0f}  eff={row['eff_emp']:.4f}  late={1-row['mean_completed']:.4f}")

if not any_plotted:
    raise ValueError("No data matched any SWEEP_CONFIGS entry.")

# ── VUG bounds (optional) ─────────────────────────────────────────────────────
if SHOW_VUG_BOUNDS:
    # Use the tau/alpha values from the last config plotted (they're graph
    # properties, same for all param settings)
    tau_fine = np.linspace(0.8, agg["tau"].max() + 0.3, 200)

    ax.plot(tau_fine, 1.0 / (1.0 + tau_fine),
            color="#D65F5F", linestyle="--", linewidth=1.2, zorder=2, alpha=0.7,
            label=r"General: $\frac{1}{1+\tau}$")

    ax.plot(agg["tau"].values, agg["poa_lb_consistent"].values,
            color="#B47CC7", linestyle="-.", linewidth=1.2, marker="D",
            markersize=3, zorder=2, alpha=0.7,
            label=r"Consistent: $\frac{1}{1+\alpha^*}$")

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel(r"Information group number $\gamma(G)$")
ax.set_ylabel("Efficiency ratio")
ax.set_ylim(0.85, 1.02)

# Set x-ticks to integer τ values found across all configs
all_taus = sorted(df_algo["tau"].dropna().unique())
ax.set_xticks(all_taus)
ax.set_xticklabels([f"{int(t)}" for t in all_taus])

ax.grid(axis="both", linewidth=0.4, linestyle=":", color="0.88", zorder=0)

# Smart legend placement
ax.legend(loc="lower left", frameon=True, borderpad=0.6,
          labelspacing=0.4, handlelength=2.0)

pad = 0.3
ax.set_xlim(min(all_taus) - pad, max(all_taus) + pad)

# ── Save ──────────────────────────────────────────────────────────────────────
fig_name = f"poa_efficiency_multisweep_{TARGET_ALGO}_short.pdf"
fig.savefig(os.path.join(OUTPUT_DIR, fig_name))
fig.savefig(os.path.join(OUTPUT_DIR, fig_name.replace(".pdf", ".png")))
plt.close(fig)
print(f"\nSaved → {os.path.join(OUTPUT_DIR, fig_name)}")