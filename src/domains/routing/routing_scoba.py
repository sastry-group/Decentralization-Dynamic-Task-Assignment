import numpy as np
from scipy.stats import norm  # placeholder for Epanechnikov
from typing import Dict, List, Tuple, Any
from collections import deque
import logging

# Solver types
from solver.scoba_types import DecisionNode, OutcomeNode, SearchTree, InteractionEvent, MODE
from solver.scoba_tree_search import generate_search_tree, get_next_attempt_idx
from solver.scoba_conflict_resolution import SCoBAAlgorithm

from domains.routing.routing_types import EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, get_travel_time_estimate, epanechnikov


rng = np.random.RandomState(1345)

def delivery_util(reward: float, ie: InteractionEvent) -> float:
    return reward

def delivery_success_prob(std_scale: float, ref_time: float, ie: InteractionEvent) -> float:

    start = max(ie.timestamps[MODE.START], ref_time)
    time_remaining = ie.timestamps[MODE.FINISH] - start  # deadline - now

    # expected duration (you need a nominal estimate — e.g., model/historical)
    expected_duration = ie.travel_time  
    sigma = max(1e-9, expected_duration / std_scale)

    # travel_time = ie.timestamps[MODE.RETURN] - ie.timestamps[MODE.FINISH]
    # mean = travel_time
    # std_val = travel_time / std_scale
    # tt_dist = epanechnikov(rng, mean, std_val)
    return epanechnikov_cdf(time_remaining, expected_duration, sigma)


def epanechnikov_cdf(x: float, mean: float, scale: float) -> float:
    """
    Compute the Epanechnikov CDF at x, with given mean and scale.
    Support is [mean - sqrt(5)*scale, mean + sqrt(5)*scale].
    """
    sqrt5 = 5 ** 0.5
    a = mean - sqrt5 * scale
    b = mean + sqrt5 * scale

    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        z = (x - mean) / scale
        return 0.75 * (z / sqrt5 - (z ** 3) / (3 * sqrt5 ** 3)) + 0.5

def scoba_routing(server, routing_sim, rng: Any = None, csv_logger=None, trial_id=None, time_step=None, comms_dict=None) -> None:
    """
    Assign tasks to drones using the SC0BA conflict-based allocation algorithm.
    `server` is a RoutingAllocation, `routing_sim` a RoutingSimulator.
    """
    # 1) Group available drones by depot
    # logging.info(f"[t={server.current_time}] Starting SCoBA assignment...")
    depots: Dict[int, List[str]] = {}
    
    for dn, dp in server.agent_prop_set.items():
        if dp.at_depot:
            # logging.info(f"Drone {dn} at depot: {dp.at_depot}")
            dnum = server.agent_set[dn].depot_number
            depots.setdefault(dnum, []).append(dn)
    # logging.info(f"Drones at depot information: {depots}")

    if csv_logger:
        for depot_num, drones in depots.items():
            depot_loc = server.agent_set[drones[0]].depot_loc
            csv_logger.log("depot_drones.csv", {
                "trial": trial_id,
                "time": time_step,
                "depot_id": depot_num,
                "drone_ids": ";".join(drones),
                "lat": depot_loc.lat,
                "lon": depot_loc.lon,
                "num_drones": len(drones),
            })


    # utils
    util_fn = lambda ie: delivery_util(routing_sim.delivery_reward, ie)
    prob_fn = lambda ref, ie: delivery_success_prob(routing_sim.tt_est_std_scale, ref, ie)

    # 2) Per-depot tree search
    task_util_allocation = {}
    all_considered = {}
    for depot_num, drones in depots.items():
        depot_loc = server.agent_set[drones[0]].depot_loc   

        in_range = set()
        for pkg, pp in routing_sim.active_packages.items():
            dist = EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
            )
            if dist <= routing_sim.distance_thresh:
                approx_travel_time = get_travel_time_estimate(
                    routing_sim.halton_nn_tree,
                    depot_loc,
                    pp.delivery,
                    routing_sim.estimate_matrix,
                    routing_sim.time_scale
                )
                
                pp.approx_travel_times[depot_num] = approx_travel_time 
                in_range.add(pkg)
                if csv_logger:
                    csv_logger.log("depot_package_distances.csv", {
                        "trial": trial_id,
                        "time": time_step,
                        "depot_id": depot_num,
                        "pkg_id": pkg,
                        "pkg_lat": pp.delivery.lat,
                        "pkg_lon": pp.delivery.lon,
                        "depot_lat": depot_loc.lat,
                        "depot_lon": depot_loc.lon,
                        "distance": dist,
                        "approx_travel_time": approx_travel_time,
                    })
        

        assigned_pkgs = set()
        for dn in drones:
            choices = in_range - assigned_pkgs
            generate_search_tree(server, dn, choices, prob_fn, util_fn, 0.0)

            tree = server.agent_prop_set[dn].tree
            att, skip = tree.child_ids.get(0, (None, None))
            if att is not None and skip is not None:
                a_util = tree.nodes[att].util
                s_util = tree.nodes[skip].util
            if tree and tree.nodes:
                # logging.info(f"[Tree] Drone {dn}: Tree has {len(tree.nodes)} nodes.")

                # Log final utility per task
                task_utils = {}
                for node in tree.nodes:
                    if isinstance(node, DecisionNode) and node.attempt:
                        task = node.task_name
                        task_utils[task] = max(task_utils.get(task, float('-inf')), node.util)
                        # logging.info(f"[Tree] Drone {dn}: Task '{task}' has utility {node.util:.2f}")

                idx = get_next_attempt_idx(tree)
                if idx != -1:
                    node = tree.nodes[idx]
                    init_util = None
                    ie_summary = server.agent_prop_set[dn].interaction_events
                    for ie in ie_summary:
                        if ie.task_name == node.task_name:
                            init_util = util_fn(ie)
                            break

                    pkg, util = node.task_name, node.util
                    assigned_pkgs.add(pkg)
                    task_util_allocation[dn] = (pkg, util)
                    all_considered[dn] = choices
                    
                


    if task_util_allocation:
        coord = SCoBAAlgorithm(allocation=server, routing_sim=routing_sim)
        true_alloc = coord.coordinate_allocation(
            task_util_allocation,
            all_considered,
            sum(u for (_, u) in task_util_allocation.values()),
            prob_fn,
            util_fn,
            0.0
        )
        # logging.info(f"[Time {server.current_time}] Final SCoBA allocation: {true_alloc}")
        available_drones = list(task_util_allocation.keys())
        unused_drones = [dn for dn in available_drones if dn not in true_alloc]
        logging.info(f"[SCoBA] Unused drones at t={server.current_time}: {unused_drones}")
        
        for dn, (pkg, _) in true_alloc.items():

            if pkg in routing_sim.busy_packages:
                # logging.warning(f"Drone {dn} lost allocation for {pkg} (already taken)")
                continue
            else:
                server.agent_task_allocation[dn] = (pkg, float("inf"))
                td, rt = sample_true_delivery_return_time(
                    server.agent_task_windows[(dn, pkg)],
                    server.current_time,
                    routing_sim.tt_est_std_scale,
                    rng,
                )
                routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
                server.agent_prop_set[dn].at_depot = False
                server.agent_prop_set[dn].current_package = pkg
                
                routing_sim.busy_packages[pkg] = routing_sim.active_packages.pop(pkg)
                routing_sim.num_active_packages -= 1
                if csv_logger:
                    dp = server.agent_prop_set[dn]
                    agent = server.agent_set[dn]
                    ie = next(
                        (e for e in dp.interaction_events if e.task_name == pkg), 
                        None
                    )
                    depot_num = server.agent_set[dn].depot_number
                    approx_tt = routing_sim.busy_packages[pkg].approx_travel_times.get(depot_num)

                    csv_logger.log("drone_assignment.csv",
                    {
                        "trial": trial_id,
                        "time": time_step,
                        "time_assigned": server.current_time,
                        "drone_id": dn,
                        "depot_id": depot_num,
                        "pkg_id": pkg,
                        # "at_depot": dp.at_depot,
                        "agent_tw_earliest_time": server.agent_task_windows[(dn, pkg)][0],
                        "agent_tw_latest_time": server.agent_task_windows[(dn, pkg)][1],
                        "agent_tw_avail": server.agent_task_windows[(dn, pkg)][2],
                        # "lat": agent.depot_loc.lat,
                        # "lon": agent.depot_loc.lon,
                        "reward": delivery_util(routing_sim.delivery_reward, ie) if ie else None,
                        "true_return_time": rt,
                        "true_delivery_time": td,
                        "success_prob": delivery_success_prob(
                            routing_sim.tt_est_std_scale,
                            server.current_time,    
                            ie) if ie else None,
                        "approx_travel_time": approx_tt,
                        "true_travel_time": rt-td
                        }
                    )
        logging.info(f"[INFO] SCoBA assigned {len(true_alloc)} drones at t={server.current_time}")
    logging.info(f"[t={server.current_time}] SCoBA assignment finished with {len(server.agent_task_allocation)} total assignments.")           
        