import matplotlib.pyplot as plt
from numpy.random import default_rng
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.patches as patches



strong_colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#000000",  # black
    "#ff1493",  # deep pink
    "#00ced1",  # dark turquoise
    "#ffa07a",  # light salmon
    "#20b2aa",  # light sea green
]



def plot_initial_map(depot_locs, drones, package_dict, radius_km):
    fig, ax = plt.subplots(figsize=(12, 10))
    rng = default_rng(42) 
    legend_elements = []
    deg_per_km = 1.0 / 111.0
    radius_deg = radius_km * deg_per_km

    # Depots
    cmap_depot = cm.get_cmap('viridis', len(depot_locs))
    depot_colors = {i: cmap_depot(i) for i in range(len(depot_locs))}
    for i, loc in enumerate(depot_locs):
        circle = patches.Circle(
            (loc.lon, loc.lat),         # (x, y) = (lon, lat)
            radius=radius_deg,
            edgecolor=depot_colors[i],
            facecolor=depot_colors[i],
            linestyle='--',
            alpha=0.2,
            label='Coverage Area' if i == 0 else ""  # only one legend entry
        )
        ax.add_patch(circle)
        ax.scatter(loc.lon, loc.lat,
                color=depot_colors[i], marker='o', s=300, alpha=0.4, 
                label='Depot' if i == 0 else "")
        ax.text(loc.lon, loc.lat + 0.002, f"Depot {i+1}", color='gray', fontsize=12)
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Depot',
                                   markerfacecolor='gray', markersize=10))

    # Drones
    spread=0.003
    drone_ids = list(drones.keys())
    drone_colors = {
        drone_id: strong_colors[i % len(strong_colors)]
        for i, drone_id in enumerate(drone_ids)
    }
    
    for drone_id in drone_ids:
        drone = drones[drone_id]
        loc = drone.depot_loc
        offset_lon = rng.uniform(-spread, spread)
        offset_lat = rng.uniform(-spread, spread)

        color = drone_colors[drone_id]
        ax.scatter(loc.lon + offset_lon, loc.lat + offset_lat,
                   color=color, marker='^', s=60)
        ax.text(loc.lon + offset_lon + 0.0002, loc.lat + offset_lat + 0.0002,
                drone_id, color=color, fontsize=9)

        # Add custom legend item
        legend_elements.append(Line2D([0], [0], marker='^', color='w', label=drone_id,
                                      markerfacecolor=color, markersize=8))

    # Packages
    for name, pkg in package_dict.items():
        loc = pkg.delivery
        ax.scatter(loc.lon, loc.lat, color='green', marker='s', s=70, label='Package' if name == list(package_dict.keys())[0] else "")
        ax.text(loc.lon + 0.001, loc.lat + 0.001, name, color='green', fontsize=9)
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Package',
                                       markerfacecolor='green', markersize=8))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend(handles=legend_elements, title="Drones", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig("results/initial_map.png")
    plt.close(fig)