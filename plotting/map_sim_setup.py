import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import cm
from numpy.random import default_rng

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# One distinct color per depot — used for depot marker, ellipse, and its drones
DEPOT_PALETTE = [
    "#4878CF",   # blue
    "#D65F5F",   # red
    "#6ACC65",   # green
    "#B47CC7",   # purple
    "#C4AD66",   # tan/gold
    "#77BEDB",   # sky blue
    "#F28E2B",   # orange
    "#59A14F",   # dark green
]

def plot_initial_map(
    city_limits,
    depots,
    drones,
    package_dict,
    radius_km,
    trial,
    time_step=0,
    show_city_box=True,
    tag="",               # e.g. "narrow", "nominal", "broad"
    show_drone_labels=False,   # turn off for clean publication figure
    show_pkg_labels=False,     # turn off when many packages
    output_dir="results",
):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    rng = default_rng(42)

    n_depots = len(depots)
    depot_color = {
        i: DEPOT_PALETTE[(i - 1) % len(DEPOT_PALETTE)]
        for i in depots
    }

    # ── Sensing ellipses ──────────────────────────────────────────────────────
    for i, depot in depots.items():
        lon = depot.location.lon
        lat = depot.location.lat
        deg_per_km_lat = 1 / 110.574
        deg_per_km_lon = 1 / (111.320 * np.cos(np.deg2rad(lat)))
        r_lat = radius_km * deg_per_km_lat
        r_lon = radius_km * deg_per_km_lon
        ellipse = patches.Ellipse(
            (lon, lat),
            width=2 * r_lon,
            height=2 * r_lat,
            edgecolor=depot_color[i],
            facecolor=depot_color[i],
            linestyle="--",
            linewidth=0.8,
            alpha=0.15,
        )
        ax.add_patch(ellipse)

    # ── Depot markers ─────────────────────────────────────────────────────────
    for i, depot in depots.items():
        lon = depot.location.lon
        lat = depot.location.lat
        color = depot_color[i]
        ax.scatter(lon, lat, color=color, marker="o", s=120,
                   zorder=5, edgecolors="white", linewidths=0.6)
        ax.text(lon + 0.001, lat + 0.0015, f"D{i}",
                color=color, fontsize=8, fontweight="500", zorder=6)

    # ── Drones — small triangles grouped at depot, no individual labels ───────
    spread = 0.0025
    drone_list = list(drones.keys())
    for drone_id in drone_list:
        drone  = drones[drone_id]
        loc    = drone.depot_loc
        dep_i  = drone.depot_number
        color  = depot_color.get(dep_i, "#888888")
        offset_lon = rng.uniform(-spread, spread)
        offset_lat = rng.uniform(-spread, spread)
        ax.scatter(
            loc.lon + offset_lon, loc.lat + offset_lat,
            color=color, marker="^", s=25,
            edgecolors="white", linewidths=0.4,
            zorder=4, alpha=0.85,
        )
        if show_drone_labels:
            ax.text(loc.lon + offset_lon + 0.0003,
                    loc.lat + offset_lat + 0.0003,
                    drone_id, color=color, fontsize=6, zorder=5)

    # ── Packages — small squares, no labels by default ────────────────────────
    pkg_names = list(package_dict.keys())
    pkg_lons  = [package_dict[n].delivery.lon for n in pkg_names]
    pkg_lats  = [package_dict[n].delivery.lat for n in pkg_names]
    ax.scatter(pkg_lons, pkg_lats,
               color="#2d6a2d", marker="s", s=18,
               edgecolors="white", linewidths=0.3,
               zorder=3, alpha=0.85, label="Package")
    if show_pkg_labels:
        for name, lon, lat in zip(pkg_names, pkg_lons, pkg_lats):
            ax.text(lon + 0.0005, lat + 0.0005, name,
                    color="#2d6a2d", fontsize=5.5, zorder=4)

    # ── City limits box ───────────────────────────────────────────────────────
    if city_limits is not None and show_city_box:
        lat_min = min(city_limits["lat_start"], city_limits["lat_end"])
        lat_max = max(city_limits["lat_start"], city_limits["lat_end"])
        lon_min = min(city_limits["lon_start"], city_limits["lon_end"])
        lon_max = max(city_limits["lon_start"], city_limits["lon_end"])
        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False, linestyle="-",
            linewidth=1.2, edgecolor="black", alpha=0.6, zorder=7,
        )
        ax.add_patch(rect)

    # ── Aspect ratio ──────────────────────────────────────────────────────────
    if city_limits is not None:
        mid_lat = 0.5 * (city_limits["lat_start"] + city_limits["lat_end"])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)), adjustable="box")

    # ── Legend — compact, one entry per depot + packages + city box ───────────
    legend_handles = []
    for i in sorted(depots.keys()):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", label=f"Depot {i}",
                   markerfacecolor=depot_color[i],
                   markeredgecolor="white", markersize=7)
        )
    # one drone entry showing depot-coloring convention
    legend_handles.append(
        Line2D([0], [0], marker="^", color="w", label="Drones (by depot)",
               markerfacecolor="gray", markeredgecolor="white", markersize=6)
    )
    legend_handles.append(
        Line2D([0], [0], marker="s", color="w", label=f"Packages (n={len(package_dict)-1})",
               markerfacecolor="#2d6a2d", markeredgecolor="white", markersize=6)
    )
    if show_city_box:
        legend_handles.append(
            Line2D([0], [0], color="black", lw=1.2, label="Sampling area")
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        borderpad=0.6,
        labelspacing=0.35,
        handletextpad=0.4,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.4, linestyle=":", color="0.80")

    # subtitle showing scenario tag
    if tag:
        ax.set_title(f"Spatial conflict: {tag}", fontsize=9, pad=4)

    suffix = f"_{tag}" if tag else ""
    fname  = f"{output_dir}/initial_map_trial_{trial}_time_{time_step}{suffix}"
    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".png")
    plt.close(fig)
    print(f"Saved → {fname}.pdf / .png")

def plot_comms_graph(comms_dict, depots, log_dir=None):

    if not comms_dict:
        num_depots = len(depots)
        comms_dict = {i: [j for j in range(1, num_depots + 1) if j != i] for i in range(1, num_depots + 1)}



    fig, ax = plt.subplots(figsize=(12, 10))
    rng = default_rng(42) 
    legend_elements = []
    deg_per_km = 1.0 / 110.2
    radius_deg = 0.01 * deg_per_km

    # Depots
    cmap_depot = cm.get_cmap('viridis', len(depots))
    depot_colors = {i: cmap_depot(i) for i in range(len(depots))}


    for i, depot in depots.items():
        lon = depot.location.lon
        lat = depot.location.lat
        circle = patches.Circle(
            (lon, lat),         # (x, y) = (lon, lat)
            radius=radius_deg,
            edgecolor=depot_colors[i-1],
            facecolor=depot_colors[i-1],
            linestyle='--',
            alpha=0.2,
            label='Coverage Area' if i-1 == 0 else ""  # only one legend entry
        )
        ax.add_patch(circle)
        ax.scatter(lon, lat,
                color=depot_colors[i-1], marker='o', s=500, alpha=0.4, 
                label='Depot' if i-1 == 0 else "")
        ax.text(lon, lat + 0.002, f"Depot {depot.depot_id}", color='gray', fontsize=12)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Depot{depot.depot_id}',
                                    markerfacecolor=depot_colors[i-1], markersize=10))


    
    # Draw communication arrows between depots
    linestyles = ['-', '--', '-.', ':']
    for src_id, targets in comms_dict.items():
        depot_loc = depots[src_id].location
        src_x = depot_loc.lon 
        src_y = depot_loc.lat 
        depot_color = depot_colors[int(src_id-1)]
        linestyle = linestyles[int(src_id - 1) % len(linestyles)]

        for tgt_id in targets:
            tgt_loc = depots[tgt_id].location
            tgt_x = tgt_loc.lon 
            tgt_y = tgt_loc.lat 

            ax.annotate(
                '', xy=(tgt_x, tgt_y), xytext=(src_x, src_y), 
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.05", linestyle=linestyle, color=depot_color, lw=2, alpha=0.6)
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend(handles=legend_elements, title="Depots", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"results/comms_graph.png")
    plt.close(fig)