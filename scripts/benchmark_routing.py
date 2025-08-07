#!/usr/bin/env python3 benchmark_routing.py
# python scripts/benchmark_routing.py   --trials 100   --timesteps 100   --n_drones 15   --n_depots 5   --new_request_prob 0.5   --time_window 30   --baseline edd   scoba_dr15_dep5_probpt5_win30_edd_test.json
import argparse
import json
import logging
import sys
import os
from pathlib import Path
import numpy as np
from sklearn.neighbors import BallTree
from numpy.random import MT19937, RandomState
# from pomdp_py.algorithms.po_uct import POUCT
from scipy.stats import uniform
import logging
import time

from csv_logger import CSVLogger


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            
sys.path.insert(0, str(ROOT / 'src'))  

from numpy.random import default_rng

# TOML parsing
import sys as _sys
if _sys.version_info >= (3, 11):
    import tomllib as _toml
else:
    import toml as _toml

def parse_city_params(toml_path: str):
    with open(toml_path, 'rb') as f:
        params = _toml.load(f)
    return {
        'lat_start': params['LATSTART'],
        'lon_start': params['LONSTART'],
        'lat_end':   params['LATEND'],
        'lon_end':   params['LONEND'],
    }



logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
from domains.routing.routing_types     import LatLonCoords, Drone, DroneProperties
from plotting.map_sim_setup import plot_initial_map

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



# Constants and file paths
PARAM_FILES = ROOT / 'param_files'
TRAVELTIME_EST = ROOT / "param_files" / "scoba_data.npz"
PARAMS_FN      = str(PARAM_FILES / 'sf_bb_params.toml')


def parse_commandline():
    p = argparse.ArgumentParser(description='Benchmark routing simulation')
    p.add_argument('--trials',           type=int,   required=True)
    p.add_argument('--n_drones',         type=int,   required=True)
    p.add_argument('--n_depots',         type=int,   required=True)
    p.add_argument('--new_request_prob', type=float, required=True)
    p.add_argument('--time_window',      type=float, required=True)
    p.add_argument('--timesteps',        type=int,   required=True)
    p.add_argument('--baseline',
                   choices=['mcts','edd','hungarian','scoba', 'ibr'],
                   required=True)
    p.add_argument('out_file_name',      type=str)
    return vars(p.parse_args())





def main():
    args = parse_commandline()
    trials = args['trials']
    results = {'trials': trials}

    # rng = default_rng(1345)
    rng = RandomState(1345)
    mcts_params = {'trials':100, 'explore':0.1, 'depth':20}

    # Precompute city distributions
    city = parse_city_params(PARAMS_FN)
    lat_dist = uniform(loc=city['lat_start'], scale=city['lat_end']-city['lat_start'])
    lon_dist = uniform(loc=city['lon_start'], scale=city['lon_end']-city['lon_start'])

    
    csv_logger = CSVLogger(filepath=f"results/logs/{args['out_file_name'].replace('.json', '')}/")
    log_dir = args['out_file_name'].replace('.json', '')
    baseline = args['baseline']
    filename = f"results/logs/{log_dir}/sim_{baseline}.log"
    logging.basicConfig(
        filename=filename,
        filemode='w',
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s:%(message)s',
    )
    
    
    logging.info(args['out_file_name'])


    # Load travel-time estimates
    arrs = np.load(str(ROOT / "param_files" / "scoba_data.npz"))
    points    = arrs["points"]       # shape (n_points, 2)
    estimates = arrs["estimates"]    # shape (n_points, n_points)
    scoba_halton_tree = BallTree(points, metric="euclidean")
    travel_time_estimates = estimates

    # Prepare drone depots
    ALL_DEPOT_LOCS = [LatLonCoords(37.774524,-122.473656),
                  LatLonCoords(37.751751,-122.410654),
                  LatLonCoords(37.718779,-122.462401),
                  LatLonCoords(37.789290,-122.426797),
                  LatLonCoords(37.739611,-122.492203)]

    DEPOT_LOCS = ALL_DEPOT_LOCS[:args['n_depots']]
    # Results containers
    late_pkgs = []
    delivered_pkgs = []
    total_pkgs = []
    in_transit_pkgs = []

    # Build drones
    n_drones = args['n_drones']
    n_depots = args['n_depots']
    drone_ordering = []
    drone_set = {}
    idx = 1
    n_per_depot = n_drones // n_depots
    remainder = n_drones % n_depots # incase odd number of drones
    for d in range(1, n_depots + 1):
        # Give one extra drone to the first `remainder` depots
        n_this_depot = n_per_depot + (1 if d <= remainder else 0)
        for _ in range(n_this_depot):
            name = f'dn{idx}'; idx += 1
            drone_ordering.append(name)
            drone_set[name] = Drone(
                drone_id=name,
                depot_number=d,
                depot_loc=DEPOT_LOCS[d - 1]
            )
    num_init = int(round(1.5 * args['n_drones'])) #request number
    

    # Run trials
    logging.info(f"Running {args['baseline']} baseline for {trials} trials.")
    if args['baseline']=='mcts':
        for i in range(trials):
                    
            logging.info(f'Trial {i+1}')
            props = {name: DroneProperties(tree=SearchTree(), interaction_events=[])
                      for name in drone_ordering}
            server = RoutingAllocation(
                agent_set=drone_set,
                agent_prop_set=props,
                agent_ordering=drone_ordering,
                max_tasks_to_consider=20,
                conflict_threshold=10
            )
            sim = setup_routing_sim(
                PARAMS_FN, scoba_halton_tree, travel_time_estimates,
                num_init_requests=num_init,
                new_request_prob=args['new_request_prob'],
                time_window_duration=args['time_window'],
                rng=rng
            )
            mdp = RoutingMCTSMDP(sim, server, args['timesteps'])
            agent = RoutingAgent(mdp)

            planner = POUCT(
                max_depth=mcts_params["depth"],
                num_sims=mcts_params["trials"],
                discount_factor=1.0,
                exploration_const=mcts_params["explore"],
            )

            # tell POUCT which rollout policy to use
            planner.set_rollout_policy(NearestPkgPolicy(mdp))

            policy = planner.plan(agent)
            for t in range(args['timesteps']):
                update_time_windows(sim, server)
                update_routing_mcts_fullstate(mdp, rng)

                actions = []
                for j in range(args['n_drones']):
                    st = get_current_routing_mcts_state(mdp, j)
                    # now call the new policy correctly:
                    a = policy.action(st)
                    mdp.drone_pkg_assignment[j] = a
                    actions.append(a)

                update_server_with_mdp_action(mdp, actions, rng)
                update_routing_sim(sim, server, rng)
            late_pkgs.append(sim.late_packages)
            delivered_pkgs.append(sim.delivered_packages)
            total_pkgs.append(sim.num_total_packages)
    else:
        # Baselines
        if args['baseline']=='edd': fn = earliest_due_date
        elif args['baseline']=='hungarian': fn = expected_hungarian
        elif args['baseline']=='scoba': fn = scoba_routing
        elif args['baseline']=='ibr': fn = iterative_best_response
        else:
            raise NotImplementedError(f"Baseline method '{args['baseline']}' is not implemented.")

        for trial in range(trials):

            props = {name: DroneProperties(tree=SearchTree(), interaction_events=[])
                      for name in drone_ordering}
            server = RoutingAllocation(agent_set=drone_set,
                                        agent_prop_set=props,
                                        agent_ordering=drone_ordering,
                                        max_tasks_to_consider=20,
                                        conflict_threshold=10)
            sim = setup_routing_sim(
                PARAMS_FN, scoba_halton_tree, travel_time_estimates,
                num_init_requests=num_init,
                new_request_prob=args['new_request_prob'],
                time_window_duration=args['time_window'],
                rng=rng,
                depot_locs=DEPOT_LOCS,
                csv_logger=csv_logger,
                in_transit_packages=in_transit_pkgs
            )
            # plot_initial_map(
            #     depot_locs=DEPOT_LOCS, drones=server.agent_set,
            #     package_dict=sim.active_packages,
            #     radius_km=sim.distance_thresh,
            #     trial=trial)
            
            timing_per_timestep = [] 
            for t in range(args['timesteps']):
                # print("time ", t)
                # logging.info(f"--- [t={t}] BEGIN TIMESTEP --- current_time = {sim.current_time}")
                update_time_windows(sim, server, csv_logger=csv_logger)
                assign = bool(sim.active_packages)
                if assign:
                    start_time = time.perf_counter()
                    fn(server, sim, rng, csv_logger=csv_logger, trial_id=trial, time_step=t)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    timing_per_timestep.append(elapsed_time)
                else:
                    logging.info(f"No active packages available. Skipping assignment.")

                update_routing_sim(sim, server, rng, csv_logger=csv_logger)
                


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
    
    with open(f"results/logs/{args['out_file_name'].replace('.json', '')}/{args['out_file_name']}", 'w') as outf:
        json.dump(results, outf, indent=2)

    csv_logger.close()

if __name__ == '__main__':
    main()
