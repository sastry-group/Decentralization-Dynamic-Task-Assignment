import matplotlib.pyplot as plt
import numpy as np
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



def plot_initial_map(city_limits, depots, drones, package_dict, radius_km, trial, show_city_box=True):

    fig, ax = plt.subplots(figsize=(12, 10))
    rng = default_rng(42) 
    legend_elements = []
    # deg_per_km = 1.0 / 110.2
    # radius_deg = radius_km * deg_per_km

    # Depots
    cmap_depot = cm.get_cmap('viridis', len(depots))
    depot_colors = {i: cmap_depot(i) for i in range(len(depots))}
    for i, depot in depots.items():
        lon = depot.location.lon
        lat = depot.location.lat
        deg_per_km_lat = 1 / 110.574
        deg_per_km_lon = 1 / (111.320 * np.cos(np.deg2rad(lat)))

        radius_lat = radius_km * deg_per_km_lat
        radius_lon = radius_km * deg_per_km_lon

        ellipse = patches.Ellipse(
            (lon, lat),
            width=2 * radius_lon,
            height=2 * radius_lat,
            edgecolor=depot_colors[i-1],
            facecolor=depot_colors[i-1],
            linestyle='--',
            alpha=0.2
        )
        ax.add_patch(ellipse)
        ax.scatter(lon, lat,
                color=depot_colors[i-1], marker='o', s=300, alpha=0.4,
                label='Depot' if i-1 == 0 else "")
        ax.text(lon, lat + 0.002, f"Depot {i}", color='gray', fontsize=12)
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

    if city_limits is not None:
        lat_start = city_limits['lat_start']
        lat_end = city_limits['lat_end']
        lon_start = city_limits['lon_start']
        lon_end = city_limits['lon_end']

        # Ensure proper ordering
        lat_min, lat_max = sorted([lat_start, lat_end])
        lon_min, lon_max = sorted([lon_start, lon_end])

        # # 1) Lock axes to the city bounds
        # ax.set_xlim(lon_min, lon_max)
        # ax.set_ylim(lat_min, lat_max)

        # 2) Draw the bounding rectangle (optional)
        if show_city_box:
            rect = patches.Rectangle(
                (lon_min, lat_min),
                width=lon_max - lon_min,
                height=lat_max - lat_min,
                fill=False,
                linestyle='-',
                linewidth=2,
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='City Limits'))

        # # Keep aspect roughly correct for degrees (lon degrees shrink with latitude)
        mid_lat = 0.5 * (lat_min + lat_max)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)), adjustable='box')
        # ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(f"results/initial_map_trial_{trial}.png", dpi=200)
        plt.close(fig)



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