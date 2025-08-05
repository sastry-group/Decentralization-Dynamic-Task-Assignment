#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
from time import perf_counter

from numpy.random import default_rng
from scipy.stats import uniform

from utils import parse_city_params
from solver.scoba_types import SearchTree


from pomdp_py import Agent
from pomdp_py.algorithms.po_uct import POUCT
from benchmark_routing import RoutingAgent

from domains.scoba_domains import (
    LatLonCoords,
    Drone,
    DroneProperties,
    RoutingAllocation,
    setup_routing_sim,
    update_time_windows,
    scoba_routing,
    expected_hungarian,
    RoutingMCTSMDP,
    update_routing_mcts_fullstate,
    NearestPkgPolicy,
    get_current_routing_mcts_state,
)

# Constants
DEPOT1 = LatLonCoords(lat=37.774524, lon=-122.473656)
DEPOT2 = LatLonCoords(lat=37.751751, lon=-122.410654)
DEPOT3 = LatLonCoords(lat=37.718779, lon=-122.462401)
DEPOT4 = LatLonCoords(lat=37.789290, lon=-122.426797)
DEPOT5 = LatLonCoords(lat=37.739611, lon=-122.492203)
DEPOTLOCS = [DEPOT1, DEPOT2, DEPOT3, DEPOT4, DEPOT5]
TRAVELTIME_EST = "./param_files/sf_halton_tt_estimates_scoba.jld2"
PARAMS_FN = "./param_files/sf_bb_params.toml"
rng = default_rng(1345)
mcts_params = {"trials": 100, "explore": 0.1, "depth": 20}


def parse_args():
    parser = argparse.ArgumentParser(description="Scaling routing computation time benchmark")
    parser.add_argument("n_depots", type=int, help="Number of depots")
    parser.add_argument("n_drones", type=int, help="Total number of drones")
    parser.add_argument("n_requests", type=int, help="Number of initial requests per trial")
    parser.add_argument("outfile", type=str, help="Output JSON filename for timing results")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials (excluding warm-up)")
    parser.add_argument("--method", choices=["scoba", "hungarian", "mcts"], default="scoba", help="Routing method to benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    comp_times = []

    # Load travel time data
    data = load_jld2(TRAVELTIME_EST)
    scoba_halton_tree = data['scoba_halton_tree']
    travel_time_estimates = data['travel_time_estimates']

    # City parameters (Python binding)
    city_params = parse_city_params(PARAMS_FN)
    lat_dist = uniform(loc=city_params["lat_start"],
                       scale=city_params["lat_end"] - city_params["lat_start"])
    lon_dist = uniform(loc=city_params["lon_start"],
                       scale=city_params["lon_end"] - city_params["lon_start"])

    total_runs = args.n_trials + 1
    for run in range(total_runs):
        # Setup drones
        drones_per_depot = args.n_drones // args.n_depots
        drone_ordering = []
        drone_set = {}
        count = 1
        for depot_idx in range(1, args.n_depots + 1):
            for _ in range(drones_per_depot):
                name = f"dn{count}"
                count += 1
                drone = Drone(depot_number=depot_idx, depot_loc=DEPOTLOCS[depot_idx-1])
                drone_ordering.append(name)
                drone_set[name] = drone

        # Drone properties
        drone_prop_set = {nm: DroneProperties(
                    tree=SearchTree(),
                    interaction_events=[])
            for nm in drone_ordering}
        drone_server = RoutingAllocation(
            agent_set=drone_set,
            agent_prop_set=drone_prop_set,
            agent_ordering=drone_ordering,
            max_tasks_to_consider=20,
            conflict_threshold=20
        )

        sim = setup_routing_sim(
            PARAMS_FN,
            scoba_halton_tree,
            travel_time_estimates,
            num_init_requests=args.n_requests,
            new_request_prob=1.0,
            rng=rng
        )

        if args.method == "scoba":
            update_time_windows(sim, drone_server)
            start = perf_counter()
            scoba_routing(drone_server, sim, rng)
            elapsed = perf_counter() - start
            comp_times.append(elapsed)

        elif args.method == "hungarian":
            update_time_windows(sim, drone_server)
            start = perf_counter()
            expected_hungarian(drone_server, sim, rng)
            elapsed = perf_counter() - start
            comp_times.append(elapsed)

        elif args.method == "mcts":
            mdp = RoutingMCTSMDP(sim, drone_server, 100, [], [])
            nearestpol = NearestPkgPolicy(mdp)
            agent  = RoutingAgent(mdp, rollout_policy=NearestPkgPolicy(mdp))
            planner = POUCT(
                max_depth=mcts_params["depth"],
                num_sims=mcts_params["trials"],
                discount_factor=1.0,
                exploration_const=mcts_params["explore"],
            )
            # mcts = MCTSSolver(
            #     n_iterations=mcts_params['trials'],
            #     depth=mcts_params['depth'],
            #     rng=rng,
            #     exploration_constant=mcts_params['explore'],
            #     reuse_tree=False,
            #     estimate_value=RolloutEstimator(nearestpol)
            # )
            # mcts_policy = POMDPs.solve(mcts, mdp)
            mcts_policy = planner.plan(agent)

            update_time_windows(sim, drone_server)
            update_routing_mcts_fullstate(mdp, rng)

            total_action_time = 0.0
            for idx in range(args.n_drones):
                state = get_current_routing_mcts_state(mdp, idx)
                start = perf_counter()
                action = mcts_policy.action(state)
                elapsed = perf_counter() - start
                mdp.drone_pkg_assignment[idx] = action
                total_action_time += elapsed
            comp_times.append(total_action_time)

    # Discard warm-up
    times = comp_times[1:]
    mean_val = sum(times) / len(times)
    sem = (sum((t - mean_val)**2 for t in times) / len(times))**0.5 / (len(times)**0.5)
    median = sorted(times)[len(times)//2]

    results = {
        "mean": mean_val,
        "sem": sem,
        "median": median
    }

    with open(args.outfile, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
