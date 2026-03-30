"""
Generate a LaTeX table comparing theoretical PoA bounds with
empirical IBR performance for each communication graph topology.

Outputs:
  1. Prints the table to stdout 
  2. Saves a CSV for reference
  3. Saves a .tex file with the table

Reads the same JSON result files as the plotting scripts.
"""

import re, json, os, glob
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to match your setup
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = "results/logs/4Paper_PoA"
OUTPUT_DIR  = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIX_PROB    = 1.0
FIX_WIN     = 30
TARGET_ALGO = "ibr"
DEPOT_ORDER = "random"       # or None to average over all orderings
FULL_COMMS_KEY = "full"

# Display labels for comms topologies (add yours here)
COMMS_LABELS = {
    "full":        "Full",
    "rm_12":       "1 dir. edge rm.",
    "rm_12_31":    "2 dir. edges rm.",
    "rm_12_31_43": "3 dir. edges rm.",
    "rm_12_31_43_54": "4 dir. edges rm.",
    "rm_12_31_43_54_51": "5 dir. edges rm.",

    "star":        "Star",
    "ring":        "Ring",
    "chain":       "Chain",
    "none":        "None (isolated)",
    "brm_12":      "1 bidir. edge rm.",
    "brm_12_45":   "2 bidir. edges rm.",
}

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
    meta["mean_late"]      = frac_late.mean()
    meta["sem_late"]       = frac_late.std(ddof=1) / np.sqrt(len(frac_late)) if len(frac_late) > 1 else 0.0

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

# ── Filter ────────────────────────────────────────────────────────────────────
mask = (
    (df["algo"].str.lower() == TARGET_ALGO) &
    np.isclose(df["prob"], FIX_PROB) &
    (df["win"] == FIX_WIN)
)
if DEPOT_ORDER is not None:
    mask = mask & (df["dporder"] == DEPOT_ORDER)

df_target = df[mask].copy()

if df_target.empty:
    raise ValueError(f"No data after filtering. Check config values.")

# ── Full-comm baseline ────────────────────────────────────────────────────────
full_rows = df_target[df_target["comms_type"] == FULL_COMMS_KEY]
if full_rows.empty:
    raise ValueError(f"No full-comm baseline found.")

full_completion = full_rows["mean_completed"].mean()

# ── Aggregate by comms_type ───────────────────────────────────────────────────
agg = df_target.groupby("comms_type").agg(
    mean_completed=("mean_completed", "mean"),
    sem_completed=("sem_completed", "mean"),
    mean_late=("mean_late", "mean"),
    sem_late=("sem_late", "mean"),
    tau=("tau", "first"),
    alpha_star=("alpha_star", "first"),
    alpha=("alpha", "first"),
    poa_lb_general=("poa_lb_general", "first"),
    poa_lb_consistent=("poa_lb_consistent", "first"),
    poa_ub=("poa_ub", "first"),
).reset_index()

agg["eff_emp"]     = agg["mean_completed"] / full_completion
agg["eff_emp_sem"] = agg["sem_completed"] / full_completion
agg = agg.sort_values("tau").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# Print summary to terminal
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print(f"  THEORETICAL vs. EMPIRICAL  —  {TARGET_ALGO.upper()},  p={FIX_PROB}, w={FIX_WIN}")
print(f"  Full-comm completion rate: {full_completion:.4f}")
print("="*90)
header = f"{'Topology':22s} {'τ':>3s} {'α*(Ḡ)':>7s} {'α(Ḡ)':>6s} │ {'1/(1+τ)':>8s} {'1/(1+α*)':>9s} {'1/α':>6s} │ {'Eff_emp':>8s} {'Late%':>7s}"
print(header)
print("─" * len(header))
for _, row in agg.iterrows():
    label = COMMS_LABELS.get(row["comms_type"], row["comms_type"])
    eff_str = f"{row['eff_emp']:.4f}" if row["comms_type"] != FULL_COMMS_KEY else "1.0000"
    print(
        f"  {label:20s} {row['tau']:3.0f} {row['alpha_star']:7.2f} {row['alpha']:6.0f} │ "
        f"{row['poa_lb_general']:8.4f} {row['poa_lb_consistent']:9.4f} {row['poa_ub']:6.4f} │ "
        f"{eff_str:>8s} {row['mean_late']*100:6.2f}%"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Save CSV
# ══════════════════════════════════════════════════════════════════════════════
csv_cols = ["comms_type", "tau", "alpha_star", "alpha",
            "poa_lb_general", "poa_lb_consistent", "poa_ub",
            "eff_emp", "eff_emp_sem", "mean_late", "sem_late", "mean_completed"]
csv_path = os.path.join(OUTPUT_DIR, f"poa_table_{TARGET_ALGO}_prob{FIX_PROB}_win{FIX_WIN}.csv")
agg[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
print(f"\nCSV saved → {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Generate LaTeX table
# ══════════════════════════════════════════════════════════════════════════════
tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Theoretical PoA bounds vs.\ empirical IBR efficiency ratio for each directed communication graph topology. "
                 r"$\mathrm{Eff}_{\mathrm{emp}}$ is the ratio of IBR completion rate under graph~$G$ to its completion rate under full communication. "
                 r"The empirical efficiency exceeds the worst-case bounds in all cases.}")
tex_lines.append(r"\label{tab:poa-comparison}")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l c c c c c c c}")
tex_lines.append(r"\toprule")
tex_lines.append(
    r"Topology & $\tau(G)$ & $\alpha^*(\bar{G})$ & $\alpha(\bar{G})$ & "
    r"$\frac{1}{1{+}\tau}$ & $\frac{1}{1{+}\alpha^*}$ & $\frac{1}{\alpha}$ & "
    r"$\mathrm{Eff}_{\mathrm{emp}}$ \\"
)
tex_lines.append(r"\midrule")

for _, row in agg.iterrows():
    label = COMMS_LABELS.get(row["comms_type"], row["comms_type"])
    tau_str   = f"{int(row['tau'])}"
    astar_str = f"{row['alpha_star']:.1f}" if row['alpha_star'] == int(row['alpha_star']) else f"{row['alpha_star']:.2f}"
    alpha_str = f"{int(row['alpha'])}"
    lb_gen    = f"{row['poa_lb_general']:.3f}"
    lb_con    = f"{row['poa_lb_consistent']:.3f}"
    ub_str    = f"{row['poa_ub']:.3f}"

    if row["comms_type"] == FULL_COMMS_KEY:
        eff_str = "1.000"
    else:
        eff_str = f"{row['eff_emp']:.3f}"

    tex_lines.append(
        f"  {label} & {tau_str} & {astar_str} & {alpha_str} & "
        f"{lb_gen} & {lb_con} & {ub_str} & {eff_str} \\\\"
    )

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

tex_str = "\n".join(tex_lines)

# Print to stdout
print("\n" + "="*70)
print("LATEX TABLE (copy into your .tex file)")
print("="*70)
print(tex_str)

# Save to file
tex_path = os.path.join(OUTPUT_DIR, f"poa_table_{TARGET_ALGO}_prob{FIX_PROB}_win{FIX_WIN}.tex")
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"\nLaTeX saved → {tex_path}")