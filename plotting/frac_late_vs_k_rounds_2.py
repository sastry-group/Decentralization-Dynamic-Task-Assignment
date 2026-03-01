import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

base_dir = "results/logs"
csv_name = "computational_efficiency_metrics.csv"




# --- folder name parser (matches your new format) ---
# FOLDER_RE = re.compile(
#     r"^dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)__probpt(?P<prob>\d+)_win(?P<win>\d+)"
#     r"_(?P<algo>[^_]+)_comms-(?P<commsk>\d+)_(?P<comms>[^_]+)_init-(?P<init>[^_]+)_dporder-(?P<dporder>[^_]+)$"
# )

FOLDER_RE = re.compile(
    r"^dr(?P<dr>\d+)_dep(?P<dep>\d+)_pkgnum(?P<pkg>\d+)__probpt(?P<prob>\d+)_win(?P<win>\d+)"
    r"_(?P<algo>[^_]+)_comms-(?P<commsk>\d+)_(?P<comms>.+?)_init-(?P<init>[^_]+)_dporder-(?P<dporder>[^_]+)$"
)

def comms_color_key(comms: str) -> str:
    if comms.startswith("edge_rm"):
        return "edge_rm"
    return comms

def parse_folder_name(folder: str):
    m = FOLDER_RE.match(folder)
    return m.groupdict() if m else None

def label_from_meta(meta: dict):
    # legend based on comms if full, init if greedy, drone order ascending
    # keep it short but informative:
    return f"{meta['comms']} | init={meta['init']} | order={meta['dporder']}"

def find_matching_folders(base_dir: str, require: dict):
    """
    require: dict of fixed fields to match, e.g.
      {"dr":"15","dep":"5","pkg":"22","prob":"05","win":"15","algo":"ibr","commsk":"5"}
    any missing key is treated as 'don't care'.
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
    # stable order
    out.sort(key=lambda x: (x[1]["comms"], x[1]["init"], x[1]["dporder"], x[0]))
    return out

def load_metrics(folder_path: str, folder: str):
    """
    Returns:
      x_total_krounds_mean_over_trials,
      y_late_mean, y_late_std,
      plus dict of extra aggregated metrics from CSV (mean across trials of sum over time)
    """
    # --- CSV aggregate: per trial sum over time, then mean across trials ---
    csv_path = os.path.join(folder_path, csv_name)
    extra = {}

    x_total_krounds = np.nan
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {"trial", "time"}.issubset(df.columns):
            # sum across time per trial for any numeric metric present
            numeric_cols = [c for c in df.columns if c not in ("trial", "time") and pd.api.types.is_numeric_dtype(df[c])]
            per_trial = df.groupby("trial", as_index=False)[numeric_cols].sum(numeric_only=True)

            if "k_rounds" in per_trial.columns:
                vals = per_trial["k_rounds"].to_numpy(dtype=float)
                x_total_krounds = float(np.nanmean(vals)) if vals.size else np.nan

            # store mean across trials for selected extras if present
            for col in ["changes", "candidate_evals", "agent_updates", "sum_eval_work", "avg_visible_degree",
                        "avg_eval_work", "avg_competitors_per_eval"]:
                if col in per_trial.columns:
                    extra[col] = float(np.nanmean(per_trial[col].to_numpy(dtype=float)))
        else:
            print(f"Warning: {csv_path} missing trial/time columns")
    else:
        print(f"Warning: missing {csv_path}")

    # --- JSON late fraction: per trial late/total then mean/std ---
    json_path = os.path.join(folder_path, f"{folder}.json")
    y_mean = np.nan
    y_std = 0.0
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            r = json.load(f)
        late = np.asarray(r.get("late", []), dtype=float)
        total = np.asarray(r.get("total", []), dtype=float)
        mask = (total > 0) & np.isfinite(late) & np.isfinite(total)
        if mask.any():
            frac = late[mask] / total[mask]
            y_mean = float(np.mean(frac))
            y_std = float(np.std(frac, ddof=1)) if len(frac) > 1 else 0.0
    else:
        print(f"Warning: missing {json_path}")

    return x_total_krounds, y_mean, y_std, extra



# --- choose scenario to plot (fixed filters) ---
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



unique_comms = sorted({meta["comms"] for _, meta in runs})
cmap = plt.get_cmap("tab10")  # or "tab20" if many cases

COMMS_COLORS = {
    comms: cmap(i % cmap.N)
    for i, comms in enumerate(unique_comms)
}

# Build "algos" dynamically as keys = folder names (or comms types if you prefer)
series = []
for folder, meta in runs:
    series.append((folder, meta))

# Optional: color by comms type (keeps plotting unchanged)
# colors = {
#     "full": "#1f77b4",
#     "ring": "#ff7f0e",
#     "none": "#2ca02c",
# }
# colors = {
#     "greedy": "#1f77b4",
#     "random": "#ff7f0e",
#     "empty": "#2ca02c",

# }

x_sum = {}
y_mean = {}
y_std = {}
labels = {}
line_colors = {}

for folder, meta in series:
    folder_path = os.path.join(base_dir, folder)

    xk, ym, ys, extra = load_metrics(folder_path, folder)

    # store as 1-point arrays to keep your plotting logic the same
    x_sum[folder] = [xk]
    y_mean[folder] = [ym]
    y_std[folder]  = [ys]

    labels[folder] = label_from_meta(meta)
    line_colors[folder] = COMMS_COLORS.get(meta["comms"], None)

# --- Plot (UNCHANGED STYLE) ---
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for folder, meta in series:
    xs = np.asarray(x_sum[folder], dtype=float)
    ys = np.asarray(y_mean[folder], dtype=float)
    yerr = np.asarray(y_std[folder], dtype=float)

    ax.errorbar(
        xs, ys,
        yerr=yerr, xerr=None,
        marker="o",
        linestyle="-" if len(xs) > 1 else "None",
        linewidth=1, markersize=4,
        color=line_colors.get(folder),
        label=labels[folder]
    )

ax.set_xlabel("Total k_round steps (sum across steps, mean across trials)")
ax.set_ylabel("Fraction of late packages (mean across trials)")
ax.grid(True, linestyle="--", alpha=0.5)

# Pull title fields from first meta
m0 = series[0][1]
ax.set_title(f"{m0['dr']} drones, {m0['dep']} depots, win {m0['win']} (pkgnum={m0['pkg']})")

plt.tight_layout()
ax.legend()

os.makedirs("results", exist_ok=True)
out_png = f"results/late_vs_kround_dr{m0['dr']}_dep{m0['dep']}_win{m0['win']}_pkg{m0['pkg']}.png"
plt.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved {out_png}")


# Load extras into dict-of-arrays (same 1-point style)
extras_by_folder = {}
for folder, meta in series:
    folder_path = os.path.join(base_dir, folder)
    xk, ym, ys, extra = load_metrics(folder_path, folder)
    extras_by_folder[folder] = extra

# choose which x-metrics you want vs late fraction
x_metrics = [
    ("sum_eval_work", "Total eval work (sum over time, mean over trials)"),
    ("candidate_evals", "Candidate evals (sum over time, mean over trials)"),
    ("changes", "Agent changes (sum over time, mean over trials)"),
    ("agent_updates", "Agent updates (sum over time, mean over trials)"),
    ("avg_visible_degree", "Avg visible degree (mean over trials)"),
    ("avg_competitors_per_eval", "Avg competitors per eval (mean over trials)"),
    ("avg_eval_work", "Avg eval work per candidate (mean over trials)"),
]

for metric, xlabel in x_metrics:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for folder, meta in series:
        xm = extras_by_folder[folder].get(metric, np.nan)
        ys = np.asarray(y_mean[folder], dtype=float)
        yerr = np.asarray(y_std[folder], dtype=float)

        ax.errorbar(
            np.asarray([xm], dtype=float), ys,
            yerr=yerr, xerr=None,
            marker="o",
            linestyle="-" if 1 > 1 else "None",
            linewidth=1, markersize=4,
            color=line_colors.get(folder),
            label=labels[folder]
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of late packages (mean across trials)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"{m0['dr']} drones, {m0['dep']} depots, win {m0['win']} (pkgnum={m0['pkg']})")
    plt.tight_layout()
    ax.legend()

    out_png = f"results/late_vs_{metric}_dr{m0['dr']}_dep{m0['dep']}_win{m0['win']}_pkg{m0['pkg']}.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")
