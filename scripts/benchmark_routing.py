#!/usr/bin/env python3 benchmark_routing.py
# python scripts/benchmark_routing.py   --trials 100   --timesteps 100   --n_drones 15   --n_depots 5   --new_request_prob 0.5   --time_window 30   --baseline edd   scoba_dr15_dep5_probpt5_win30_edd_test.json
import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import BallTree
# from pomdp_py.algorithms.po_uct import POUCT
import time

from csv_logger import CSVLogger


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            
sys.path.insert(0, str(ROOT / 'src'))  



# TOML parsing

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:
    import toml as _toml


logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
from domains.routing.routing_types     import LatLonCoords, Drone, DroneProperties, Depot
from plotting.map_sim_setup import plot_initial_map, plot_comms_graph

from domains.routing.routing_simulator import (
    setup_routing_sim,
    update_time_windows,
    update_routing_sim,
)

from domains.routing.routing_baselines import expected_hungarian, earliest_due_date, get_current_routing_mcts_state
from domains.routing.mcts import (
    RoutingAgent,
    RoutingMCTSMDP,
    NearestPkgPolicy,
    update_server_with_mdp_action,
    update_routing_mcts_fullstate
)


from domains.routing.routing_scoba import scoba_routing
from domains.routing.routing_ibr import iterative_best_response
from solver.scoba_types                 import GenericAllocation as RoutingAllocation
from solver.scoba_types import SearchTree
from domains.routing.travel_model import initialize_travel_model
from domains.routing.graph_builder import (
    build_comms_graph,
    comms_dict_from_graph,
    graph_metrics,
)


# Constants and file paths
PARAM_FILES = ROOT / "param_files"
TRAVELTIME_EST = PARAM_FILES / "scoba_data.npz"
PARAMS_BY_DEPOTS = {
    2: str(PARAM_FILES / "sf_bb_params_2dpts_test.toml"),
    5: str(PARAM_FILES / "sf_bb_params.toml"),
    6: str(PARAM_FILES / "sf_bb_params_more_overlap.toml"),
    10: str(PARAM_FILES / "sf_bb_params_10dpts.toml"),
}

DEPOT_COORDS = [
    (37.774524, -122.473656),
    (37.751751, -122.410654),
    (37.718779, -122.462401),
    (37.789290, -122.426797),
    (37.739611, -122.492203),
]



def parse_city_params(toml_path: str):
    with open(toml_path, 'rb') as f:
        params = _toml.load(f)
    return {
        'lat_start': params['LATSTART'],
        'lon_start': params['LONSTART'],
        'lat_end':   params['LATEND'],
        'lon_end':   params['LONEND'],
    }

def generate_depot_coords(n_depots: int,
                          center=(37.76, -122.45),
                          spacing=0.03):
    """
    Generate depot coordinates on a grid around a center.
    spacing ~ 0.03 ≈ 3km
    """
    coords = []
    side = int(np.ceil(np.sqrt(n_depots)))
    lat0, lon0 = center

    for i in range(side):
        for j in range(side):
            if len(coords) >= n_depots:
                break
            lat = lat0 + (i - side // 2) * spacing
            lon = lon0 + (j - side // 2) * spacing
            coords.append((lat, lon))

    return coords

def build_depots(n_depots: int) -> dict[int, Depot]:
    if n_depots <= len(DEPOT_COORDS):
        coords = DEPOT_COORDS[:n_depots]
    else:
        coords = generate_depot_coords(n_depots)

    return {
        i + 1: Depot(
            depot_id=f"dp{i+1}",
            location=LatLonCoords(lat=coords[i][0], lon=coords[i][1]),
            capacity=5,
        )
        for i in range(n_depots)
    }

def build_drones(n_drones: int, depots: dict[int, Depot]):
    n_depots = len(depots)
    drone_ordering = []
    drone_set = {}

    n_per = n_drones // n_depots
    rem = n_drones % n_depots
    idx = 1

    for depot_idx, depot in depots.items():
        n_this = n_per + (1 if depot_idx <= rem else 0)
        for _ in range(n_this):
            name = f"dn{idx}"
            idx += 1
            drone_ordering.append(name)
            drone_set[name] = Drone(
                drone_id=name,
                depot_number=depot_idx,
                depot_loc=depot.location,
            )

    return drone_ordering, drone_set



def comms_graph_from_mode(mode: str, n_depots: int):
    G = build_comms_graph(n_depots, mode)
    metrics = graph_metrics(G)
    comms_dict = comms_dict_from_graph(G)
    return comms_dict, metrics




def parse_commandline():
    p = argparse.ArgumentParser(description="Benchmark routing simulation")

    # core sim
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--n_drones", type=int, default=6)   
    p.add_argument("--n_depots", type=int, default=2)
    p.add_argument("--num-init-requests", type=int, default=None, help="Override initial requests. Default is round(1.5 * n_drones).")
    p.add_argument("--new_request_prob", type=float, default=0.5)
    p.add_argument("--time_window", type=int, default=15)
    p.add_argument("--seed", type=int, default=1345)

    # baseline
    p.add_argument("--baseline", choices=["edd", "hungarian", "scoba", "ibr"], default="ibr")

    # params selection
    p.add_argument("--params-file", type=str, default=None,
                   help="Override TOML param file. If omitted, chosen by --n_depots.")

    # IBR options
    p.add_argument("--init_method", choices=["greedy", "random", "empty"], default="greedy")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Allow overlapping claims (race at arrival). Default: False.")
    p.add_argument("--dynamic_tasks", action="store_true",
                   help="Whether to allow new tasks to arrive during the simulation. Default: False.")

    p.add_argument("--comms_mode", type=str, default="full")        
    p.add_argument("--depot-order", type=str, default="asc",
                choices=["asc", "desc", "random"])
    p.add_argument("--plot-comms", action="store_true")
    p.add_argument("--plot-init", action="store_true")
    # logging verbosity
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")

    return vars(p.parse_args())




def main():
    args = parse_commandline()

    # choose params file
    if args["params_file"] is not None:
        params_fn = args["params_file"]
    else:
        if args["n_depots"] not in PARAMS_BY_DEPOTS:
            raise ValueError(
                f"No default params file for n_depots={args['n_depots']}. "
                f"Use --params-file to override."
            )
        params_fn = PARAMS_BY_DEPOTS[args["n_depots"]]
    
    # rng = np.random.default_rng(args["seed"])
    base_seed = args["seed"]
    comms_dict, comms_metrics = comms_graph_from_mode(args["comms_mode"], args["n_depots"])

    trials = args['trials']
    results = {'trials': trials}
    mcts_params = {'trials':100, 'explore':0.1, 'depth':20}
    # logging + output dir
    
    baseline = args["baseline"]
    comms_tag = args["comms_mode"]
    init_tag  = args["init_method"]
    order_tag = args["depot_order"]

    num_init = args["num_init_requests"]
    if num_init is None:
        num_init = int(round(1.5 * args["n_drones"]))

    overlap = "overlap" if args["allow_overlap"] else "nooverlap"
    dynamic_tasks = "dynamic" if args["dynamic_tasks"]  else "static"

    log_dir = (
        f"dr{args['n_drones']}_dep{args['n_depots']}_pkgnum{num_init}"
        f"_ntrials{trials}"
        f"_nsteps{args['timesteps']}"
        f"_probpt{str(args['new_request_prob']).replace('.', '')}"
        f"_win{args['time_window']}"
        f"_{args['baseline']}"
        f"_comms-{comms_tag}"
        f"_init-{init_tag}"
        f"_dporder-{order_tag}"
        f"_overlap-{overlap}"
        f"_tasks-{dynamic_tasks}"
        # f"_darr_narrow"
    )
    out_dir = ROOT / "results" / "logs" / log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(filepath=str(out_dir) + "/")

    
    logging.basicConfig(
        filename=str(out_dir / f"sim_{baseline}.log"),
        filemode="w",
        level=getattr(logging, args["log_level"]),
        format="[%(asctime)s] %(levelname)s:%(message)s",
    )
    logging.info(f"log_dir={log_dir}")
    logging.info(f"params_fn={params_fn}")
    logging.info(f"comms_dict={comms_dict}")
    
    metrics_d = comms_metrics.to_dict()
    logging.info(f"comms_metrics={metrics_d}")
    print(
        f"  Comms graph: mode={comms_tag}  n_depots={args['n_depots']}\n"
        f"    τ(G)={comms_metrics.tau}  "
        f"α*(Ḡ)={comms_metrics.alpha_star_recip:.2f}  "
        f"α(Ḡ)={comms_metrics.alpha_recip}\n"
        f"    PoA bounds:  general ≥ {comms_metrics.poa_lb_general:.4f}  "
        f"consistent ≥ {comms_metrics.poa_lb_consistent:.4f}  "
        f"upper ≤ {comms_metrics.poa_ub:.4f}"
    )
    # travel-time estimates
    arrs = np.load(str(TRAVELTIME_EST))
    points = arrs["points"]
    estimates = arrs["estimates"]

    INVALID = 100000

    valid_nodes = np.all(estimates < INVALID, axis=1)

    print("Original Halton nodes:", len(points))
    print("Valid Halton nodes:", np.sum(valid_nodes))

    # filter points and matrix
    points = points[valid_nodes]
    estimates = estimates[valid_nodes][:, valid_nodes]

    # rebuild tree with filtered nodes
    scoba_halton_tree = BallTree(points, metric="euclidean")

    travel_time_estimates = estimates
    initialize_travel_model(scoba_halton_tree, travel_time_estimates, time_scale=60.0)




    # world
    depots = build_depots(args["n_depots"])
    drone_ordering, drone_set = build_drones(args["n_drones"], depots)

    allow_overlap = bool(args["allow_overlap"])
    dynamic_tasks = bool(args["dynamic_tasks"])

    if args["plot_comms"]:
        plot_comms_graph(comms_dict, depots=depots, log_dir=log_dir)

    # choose baseline function
    if baseline == "edd":
        fn = earliest_due_date
    elif baseline == "hungarian":
        fn = expected_hungarian
    elif baseline == "scoba":
        fn = scoba_routing
    elif baseline == "ibr":
        fn = iterative_best_response
    else:
        raise NotImplementedError(baseline)


    # Results 
    late_pkgs = []
    delivered_pkgs = []
    total_pkgs = []
    in_transit_pkgs = []


    # Run trials
    logging.info(f"Running {args['baseline']} baseline for {trials} trials.")
   

    for trial in range(trials):
        trial_seed = base_seed + trial
        rng = np.random.default_rng(trial_seed)   
        task_rng = np.random.default_rng(trial_seed + 9999) # fixing task rng
        print(f"Trial {trial+1}/{trials}")

        props = {name: DroneProperties(tree=SearchTree(), interaction_events=[])
                    for name in drone_ordering}
        server = RoutingAllocation(agent_set=drone_set,
                                    agent_prop_set=props,
                                    agent_ordering=drone_ordering,
                                    max_tasks_to_consider=20,
                                    conflict_threshold=20)



        sim = setup_routing_sim(
            server,
            params_fn,
            scoba_halton_tree,
            travel_time_estimates,
            num_init_requests=num_init,
            new_request_prob=args["new_request_prob"],
            time_window_duration=args["time_window"],
            rng=task_rng,
            depots=depots,
            csv_logger=csv_logger,
            in_transit_packages=None,
        )

        if args["plot_init"]:
            city = parse_city_params(params_fn)
            plot_initial_map(
                city,
                depots=depots,
                drones=server.agent_set,
                package_dict=sim.active_packages,
                radius_km=sim.distance_thresh,
                trial=trial,
                time_step = 0
            )
        
        timing_per_timestep = [] 
        for t in range(args['timesteps']):
            # print("time ", t)
            # logging.info(f"--- [t={t}] BEGIN TIMESTEP --- current_time = {sim.current_time}")
            update_time_windows(sim, server, csv_logger=csv_logger)
            if sim.active_packages:
                start = time.perf_counter()

                if baseline == "ibr":
                    fn(
                        server, sim, rng,
                        csv_logger=csv_logger,
                        init_method=args["init_method"],
                        trial_id=trial,
                        time_step=t,
                        comms_dict=comms_dict,
                        allow_overlap=allow_overlap,
                        depot_order=args["depot_order"],
                    )
                else:
                    fn(
                        server, sim, rng,
                        csv_logger=csv_logger,
                        trial_id=trial,
                        time_step=t,
                        comms_dict=comms_dict,
                        allow_overlap=allow_overlap,
                        depot_order=args["depot_order"],
                    )

                timing_per_timestep.append(time.perf_counter() - start)
            else:
                logging.info("No active packages. Skipping assignment.")

            update_routing_sim(trial, sim, server, rng, csv_logger=csv_logger, allow_overlap=allow_overlap, dynamic_tasks=dynamic_tasks)
            if sim.new_packages_created and args["plot_init"]:
                plot_initial_map(
                    city,
                    depots=depots,
                    drones=server.agent_set,
                    package_dict=sim.active_packages,
                    radius_km=sim.distance_thresh,
                    trial=trial, time_step=t,
                )
                sim.new_packages_created = False
            


        late_pkgs.append(sim.late_packages)
        in_transit_pkgs.append(len(sim.busy_packages))
        delivered_pkgs.append(sim.delivered_packages)
        total_pkgs.append(sim.num_total_packages)
            

        

    # Output
    if timing_per_timestep:
        avg_time = sum(timing_per_timestep) / len(timing_per_timestep)
        results['avg_time_per_assignment_step_sec'] = avg_time
    results['late'] = late_pkgs
    results['in_transit'] = in_transit_pkgs
    results['delivered'] = delivered_pkgs
    results['total'] = total_pkgs
    results['comms_metrics'] = comms_metrics.to_dict()

    with open(out_dir / f"{log_dir}.json", 'w') as outf:
        json.dump(results, outf, indent=2)

    csv_logger.close()

if __name__ == '__main__':
    main()
