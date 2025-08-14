import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import deque
import logging


from domains.routing.routing_scoba import delivery_success_prob
from solver.scoba_types import InteractionEvent, MODE, GenericAllocation as RoutingAllocation
from domains.routing.routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, get_travel_time_estimate
from domains.graph_builder import build_comm_structure



def build_comm_structure(depots: Dict[int, list], comms_dict: dict = None):
    """
    Build a mapping of which depots can see which other depots.
    comms_dict format: {depot_id: [list of visible depot_ids]}.
    If comms_dict is None, default to full communication (all-to-all).
    """
    depot_ids = list(depots.keys())
    if comms_dict is None:
        # Full graph: each depot sees all others (including itself)
        return {d: depot_ids for d in depot_ids}
    else:
        # Ensure each depot exists in the dictionary
        return {d: comms_dict.get(d, []) for d in depot_ids}



def delivery_util(std_scale, reward: float, ref_time: float, ie: InteractionEvent) -> float:
    prob_success = delivery_success_prob(
        std_scale=std_scale,
        ref_time=ref_time,
        ie=ie,
    )
    return reward * prob_success


def welfare_function(task_set: Set[str], delivery_reward=1000.0) -> float:
    unique_tasks = set(task_set)
    return sum(delivery_reward for _ in unique_tasks)

def compute_utility(ie: InteractionEvent,
                    agent: str,
                    assigned: Dict[str, str],
                    server,
                    routing_sim,
                    mode="exact",
                    lambda_conflict=500,
                    delivery_reward=1000.0):
    pkg = ie.task_name
    p_succ = delivery_success_prob(
        std_scale=routing_sim.tt_est_std_scale,
        ref_time=ie.timestamps[MODE.FINISH],
        ie=ie
    )
    base_util = delivery_reward * p_succ
    

    if mode == "approximate":
        # Only penalize if neighbors also selected this pkg
        neighbors = assigned.get("neighbors", {}).get(agent, [])
        neighbor_pkgs = {assigned[n] for n in neighbors if assigned.get(n)}
        conflict = pkg in neighbor_pkgs
        return base_util - (lambda_conflict if conflict else 0)

    elif mode == "exact":
        # Requires full visibility of all current assignments
        current_x = set(assigned.values()) - {pkg}
        welfare_with = welfare_function(current_x.union({pkg}))
        welfare_without = welfare_function(current_x)
        logging.info(f"Agent {agent}, with base util: {base_util}, welfare_with {welfare_with}, w/o {welfare_without}")
        return welfare_with - welfare_without

    else:
        raise ValueError(f"Unknown utility mode '{mode}'")

    



def delivery_success_prob(std_scale: float, ref_time: float, ie) -> float:
    # remaining time until the hard deadline
    start = max(ie.timestamps[MODE.START], ref_time)
    time_remaining = ie.timestamps[MODE.FINISH] - start  # deadline - now

    # expected duration (you need a nominal estimate — e.g., model/historical)
    expected_duration = ie.travel_time  

    # std dev as a fraction of mean (std_scale ≈ 3.0 => σ = μ/3)
    sigma = max(1e-9, expected_duration / std_scale)

    # CDF of finishing before the deadline
    return epanechnikov_cdf(time_remaining, expected_duration, sigma)


def epanechnikov_cdf(x: float, mean: float, scale: float) -> float:
    """
    Compute the Epanechnikov CDF at x, with given mean and scale.
    Support is [mean - sqrt(5)*scale, mean + sqrt(5)*scale].
    """
    sqrt5 = 5 ** 0.5
    a = mean - sqrt5 * scale
    b = mean + sqrt5 * scale

    if x <= a: return 0.0
    if x >= b: return 1.0

    u = (x - mean) / (sqrt5 * scale) # normalized to [-1,1]
    # CDF of Epanechnikov kernel on [-1,1]: 0.5 + 0.75*(u - u^3/3)
    return max(0.0, min(1.0, 0.5 + 0.75 * (u - (u**3)/3.0)))



def best_response_utility(current_time, ie, reward=1000):
    start, end, avail = ie.time_window
    delivery_est = avail  # or an estimate
    penalty = max(0, delivery_est - end)
    return reward - penalty



def iterative_best_response(server: RoutingAllocation, routing_sim: RoutingSimulator, rng=None, csv_logger=None,
                            init_method = "greedy", trial_id=None, time_step=None, comms_dict=None):
    max_iters = 40
    depots: Dict[int, List[str]] = {}
    # group drones by depot
    for dn, dp in server.agent_prop_set.items():
        if dp.at_depot:
            dnum = server.agent_set[dn].depot_number
            depots.setdefault(dnum, []).append(dn)
    # logging.info(f"Drones at depot information: {depots}")

    # depot_visibility = build_comm_structure(depots, comms_dict)
    depot_visibility = comms_dict

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
    

    # Precompute in-range packages per depot
    in_range_pkgs: Dict[int, Set[str]] = {}
    interaction_events_by_drone: Dict[str, List[InteractionEvent]] = {}

    for depot_id, drones in depots.items():
        depot_loc = server.agent_set[drones[0]].depot_loc
        in_range_pkgs[depot_id] = set()

        for pkg, pp in routing_sim.active_packages.items():
            dist = EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
            )
            if dist <= routing_sim.distance_thresh:
                approx_tt = get_travel_time_estimate(
                    routing_sim.halton_nn_tree,
                    depot_loc, pp.delivery,
                    routing_sim.estimate_matrix,
                    routing_sim.time_scale
                )
                pp.approx_travel_times[depot_id] = approx_tt
                in_range_pkgs[depot_id].add(pkg)
                if csv_logger:
                    csv_logger.log("depot_package_distances.csv", {
                        "trial": trial_id,
                        "time": time_step,
                        "depot_id": depot_id,
                        "pkg_id": pkg,
                        "pkg_lat": pp.delivery.lat,
                        "pkg_lon": pp.delivery.lon,
                        "depot_lat": depot_loc.lat,
                        "depot_lon": depot_loc.lon,
                        "distance": dist,
                        "approx_travel_time": approx_tt,
                    })

        # Precompute events for each drone in this depot
        for dn in drones:
            interaction_events_by_drone[dn] = [
                ie for ie in server.agent_prop_set[dn].interaction_events
                if ie.task_name in in_range_pkgs[depot_id]
                and ie.task_name not in routing_sim.busy_packages
            ]


    # Initial assignment per depot, sequential, no cross-depot restriction
    assigned: Dict[str, str] = {}
    best_ies: Dict[str, Tuple[str, float]] = {}
    final_assignment: Dict[str, Tuple[str, float, InteractionEvent]] = {}
    all_task_utils: Dict[Tuple[str, str], float] = {}

    for depot_id, drones in depots.items():
        assigned_pkgs_in_depot: Set[str] = set()

        if init_method == "random":
            for dn in drones:
                available = [
                    ie for ie in interaction_events_by_drone.get(dn, [])
                    if ie.task_name not in assigned_pkgs_in_depot
                ]
                if available:
                    ie = rng.choice(available)
                    assigned[dn] = ie.task_name
                    assigned_pkgs_in_depot.add(ie.task_name)

        elif init_method == "greedy":
            for dn in drones:
                best_ie = None
                best_util = float("-inf")
                for ie in interaction_events_by_drone.get(dn, []):
                    if ie.task_name in assigned_pkgs_in_depot:
                        continue
                    u = delivery_util(routing_sim.tt_est_std_scale, routing_sim.delivery_reward, ref_time=routing_sim.current_time, ie=ie)
                    logging.info(f"Timestep: {time_step}, drone {dn}, pkg {ie.task_name}, utility: {u}")
                    all_task_utils[(dn, ie.task_name)] = u
                    if u > best_util:
                        best_util = u
                        best_ie = ie
                if best_ie:
                    assigned[dn] = best_ie.task_name
                    assigned_pkgs_in_depot.add(best_ie.task_name)
                    best_ies[dn] = (best_ie.task_name, best_util)
                    final_assignment[dn] = (best_ie.task_name, best_util, best_ie)
                else:
                    best_ies[dn] = (None, 0.0)
    
    # Iterative best response
    iter_count = 0
    updated = True
    while updated and iter_count < max_iters:
        updated = False
        for dn in assigned.keys():  # all drones assigned in step 2
            if not server.agent_prop_set[dn].at_depot:
                continue
            
            depot_id = server.agent_set[dn].depot_number
            visible_depots = depot_visibility[depot_id] + [depot_id]
            visible_depots.sort()
            visible_drones = [dr for dep in visible_depots for dr in depots.get(dep, [])]
            assigned_pkgs_visible = {assigned[other] for other in visible_drones if other in assigned}

            best_ie = None
            best_util = float("-inf")
            for ie in interaction_events_by_drone.get(dn, []):
                pkg = ie.task_name
                if pkg in assigned_pkgs_visible or pkg in routing_sim.busy_packages:
                    continue
                u = compute_utility(
                    ie=ie,
                    agent=dn,
                    assigned=assigned,
                    server=server,
                    routing_sim=routing_sim,
                    mode="exact",
                    lambda_conflict=500,
                    delivery_reward=routing_sim.delivery_reward
                )
                if u > best_util:
                    best_util = u
                    best_ie = ie
            if best_ie and assigned[dn] != best_ie.task_name:
                assigned[dn] = best_ie.task_name
                updated = True
                final_assignment[dn] = (best_ie.task_name, best_util, best_ie)
        iter_count += 1





    for dn, pkg in assigned.items():
        if pkg in routing_sim.busy_packages:
            logging.warning(f"Drone {dn} lost allocation for {pkg} (already taken)")
            continue
        else:
            server.agent_task_allocation[dn] = (pkg, float('inf'))
            td, rt = sample_true_delivery_return_time(
                server.agent_task_windows[(dn, pkg)],
                server.current_time,
                routing_sim.tt_est_std_scale,
                rng
            )
            routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
            server.agent_prop_set[dn].at_depot = False
            server.agent_prop_set[dn].current_package = pkg

            routing_sim.busy_packages[pkg] = routing_sim.active_packages.pop(pkg)
            routing_sim.num_active_packages -= 1
            final_pkg, util, ie = final_assignment[dn]
            depot_num = server.agent_set[dn].depot_number
            assert final_pkg == pkg, f"Final assigned pkg mismatch for {dn}: expected {pkg}, got {final_pkg}"

            if csv_logger:
                dp = server.agent_prop_set[dn]
                csv_logger.log("drone_assignment.csv", {
                    "trial": trial_id,
                    "time": time_step,
                    "drone_id": dn,
                    "pkg_id": pkg,
                    "agent_tw_earliest_time": server.agent_task_windows[(dn, pkg)][0],
                    "agent_tw_latest_time": server.agent_task_windows[(dn, pkg)][1],
                    "agent_tw_avail": server.agent_task_windows[(dn, pkg)][2],
                    "true_return_time": rt,
                    "true_delivery_time": td,
                    "success_prob": util,
                    "approx_travel_time": routing_sim.busy_packages[pkg].approx_travel_times.get(depot_num),
                    "true_travel_time": rt-td
                })