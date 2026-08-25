"""
Fig 4: Empirical IBR efficiency vs. theoretical PoA bounds.

x-axis: τ(G)  (information group number)
y-axis: efficiency ratio

Plots:
  - IBR empirical efficiency (data points with error bars)
  - General VUG lower bound:    1 / (1 + tau)          [dashed]
  - Consistent VUG lower bound: 1 / (1 + alpha*(Gbar))      [dash-dot]
  - Upper bound (any utility):  1 / alpha(Gbar)              [dotted]
"""

import re, json, os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    "legend.fontsize":    8.5,
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
# CONFIG — edit these to match your setup
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = "results/logs/4Paper_PoA"       # folder with all runs
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed experimental conditions (must match the runs you want to compare)
FIX_PROB   = 1.0
FIX_WIN    = 30
TARGET_ALGO = "ibr"          # algorithm to compute efficiency for
DEPOT_ORDER = "random"       # pick one ordering, or set to None to average all

# Comms topologies: list ALL that appear in your data.
# The script auto-detects which is the "full" baseline.
# You can also hardcode:  FULL_COMMS_KEY = "full"
FULL_COMMS_KEY = "full"

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

# ── Load data ─────────────────────────────────────────────────────────────────
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

    with open(json_path) as f:
        r = json.load(f)

    late  = np.array(r["late"],  dtype=float)
    total = np.array(r["total"], dtype=float)
    valid = total > 0
    if not np.any(valid):
        continue

    frac_late = late[valid] / total[valid]
    frac_completed = 1.0 - frac_late  # completion rate per trial

    meta["mean_completed"] = frac_completed.mean()
    meta["sem_completed"]  = frac_completed.std(ddof=1) / np.sqrt(len(frac_completed)) if len(frac_completed) > 1 else 0.0
    meta["mean_late"]      = frac_late.mean()
    meta["sem_late"]       = frac_late.std(ddof=1) / np.sqrt(len(frac_late)) if len(frac_late) > 1 else 0.0

    # PoA metrics from comms_metrics
    cm = r.get("comms_metrics", {})
    meta["tau"]             = cm.get("tau", None)
    meta["alpha_star"]      = cm.get("alpha_star_reciprocal", None)
    meta["alpha"]           = cm.get("alpha_reciprocal", None)
    meta["poa_lb_general"]  = cm.get("poa_lb_general", None)
    meta["poa_lb_consistent"] = cm.get("poa_lb_consistent", None)
    meta["poa_ub"]          = cm.get("poa_ub", None)

    records.append(meta)

if not records:
    raise ValueError(f"No valid experiment folders found in:\n  {BASE_DIR}")

df = pd.DataFrame(records)

# ── Filter to target algo, fixed params ───────────────────────────────────────
mask = (
    (df["algo"].str.lower() == TARGET_ALGO) &
    np.isclose(df["prob"], FIX_PROB) &
    (df["win"] == FIX_WIN)
)
if DEPOT_ORDER is not None:
    mask = mask & (df["dporder"] == DEPOT_ORDER)

df_target = df[mask].copy()

if df_target.empty:
    raise ValueError(
        f"No data for algo={TARGET_ALGO}, prob={FIX_PROB}, win={FIX_WIN}, "
        f"dporder={DEPOT_ORDER}.\n"
        f"Available: algo={df['algo'].unique()}, prob={df['prob'].unique()}, "
        f"win={df['win'].unique()}, dporder={df['dporder'].unique()}"
    )

print(f"Found {len(df_target)} runs for {TARGET_ALGO}")
print(f"Comms types: {sorted(df_target['comms_type'].unique())}")
print(f"τ values:    {sorted(df_target['tau'].dropna().unique())}")

# ── Get full-comm baseline ────────────────────────────────────────────────────
full_rows = df_target[df_target["comms_type"] == FULL_COMMS_KEY]
if full_rows.empty:
    raise ValueError(
        f"No full-communication baseline found (comms_type='{FULL_COMMS_KEY}').\n"
        f"Available comms_type: {df_target['comms_type'].unique()}"
    )

full_completion = full_rows["mean_completed"].mean()
print(f"\nFull-comm completion rate: {full_completion:.4f}")

# ── Aggregate by comms_type ───────────────────────────────────────────────────
# If DEPOT_ORDER is None, we average over all depot orders per comms_type
agg = df_target.groupby("comms_type").agg(
    mean_completed=("mean_completed", "mean"),
    sem_completed=("sem_completed", "mean"),   # approximate
    mean_late=("mean_late", "mean"),
    tau=("tau", "first"),
    alpha_star=("alpha_star", "first"),
    alpha=("alpha", "first"),
    poa_lb_general=("poa_lb_general", "first"),
    poa_lb_consistent=("poa_lb_consistent", "first"),
    poa_ub=("poa_ub", "first"),
).reset_index()

# Compute empirical efficiency
agg["eff_emp"] = agg["mean_completed"] / full_completion
agg["eff_emp_sem"] = agg["sem_completed"] / full_completion  # propagated SE

# Sort by tau
agg = agg.sort_values("tau").reset_index(drop=True)

print("\n" + "="*70)
print("AGGREGATED RESULTS")
print("="*70)
for _, row in agg.iterrows():
    print(f"  {row['comms_type']:20s}  τ={row['tau']:.0f}  α*={row['alpha_star']:.1f}  "
          f"α={row['alpha']:.0f}  "
          f"Eff_emp={row['eff_emp']:.4f}  "
          f"PoA_gen≥{row['poa_lb_general']:.4f}  "
          f"PoA_con≥{row['poa_lb_consistent']:.4f}  "
          f"PoA_ub≤{row['poa_ub']:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.8))

tau_vals = agg["tau"].values

# Theoretical bounds as lines (use a fine grid for smooth curves)
tau_fine = np.linspace(max(1, tau_vals.min() - 0.3), tau_vals.max() + 0.3, 200)

# For the consistent bound, we need α* at each τ. We interpolate from data.
# But since α* depends on the specific graph (not just τ), we plot the
# actual data-point bounds, not a smooth curve.

# General VUG bound: 1/(1+τ)
ax.plot(tau_fine, 1.0 / (1.0 + tau_fine),
        color="#D65F5F", linestyle="--", linewidth=1.5, zorder=2,
        label=r"General VUG: $\frac{1}{1+\tau(G)}$")

# Consistent VUG bound: 1/(1+α*) — plot as discrete points connected
ax.plot(tau_vals, agg["poa_lb_consistent"].values,
        color="#B47CC7", linestyle="-.", linewidth=1.5, marker="D",
        markersize=4, zorder=2,
        label=r"Consistent VUG: $\frac{1}{1+\alpha^*(\bar{G})}$")

# Upper bound: 1/α — plot as discrete points
ax.plot(tau_vals, agg["poa_ub"].values,
        color="#888888", linestyle=":", linewidth=1.2, marker="x",
        markersize=5, zorder=2,
        label=r"Upper bound: $\frac{1}{\alpha(\bar{G})}$")

# Empirical IBR efficiency
ax.errorbar(tau_vals, agg["eff_emp"].values, yerr=agg["eff_emp_sem"].values,
            color="#6ACC65", marker="^", markersize=7, linewidth=1.8,
            linestyle="-", capsize=3, zorder=4,
            label=f"IBR empirical")

# Formatting
ax.set_xlabel(r"Information group number $\gamma(G)$")
ax.set_ylabel("Efficiency ratio")
ax.set_ylim(0, 1.05)
ax.set_xticks(tau_vals)
ax.set_xticklabels([f"{int(t)}" for t in tau_vals])
ax.grid(axis="both", linewidth=0.4, linestyle=":", color="0.88", zorder=0)
ax.legend(loc="lower left", frameon=True, borderpad=0.6, labelspacing=0.4)

# x padding
pad = 0.3
ax.set_xlim(tau_vals.min() - pad, tau_vals.max() + pad)

# ── Save ──────────────────────────────────────────────────────────────────────
dr  = df_target["dr"].iloc[0]
dep = df_target["dep"].iloc[0]
fig_name = f"poa_efficiency_tau_dr{dr}_dep{dep}_prob{FIX_PROB}_win{FIX_WIN}.pdf"
fig.savefig(os.path.join(OUTPUT_DIR, fig_name))
fig.savefig(os.path.join(OUTPUT_DIR, fig_name.replace(".pdf", ".png")))
plt.close(fig)
print(f"\nSaved → {os.path.join(OUTPUT_DIR, fig_name)}")