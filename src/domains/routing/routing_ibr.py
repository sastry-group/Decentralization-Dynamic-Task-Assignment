import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import deque
import logging


from solver.scoba_types import InteractionEvent, MODE, GenericAllocation as RoutingAllocation
from domains.routing.routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, get_travel_time_estimate





def create_comm_graph(depots: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """
    Create a communication graph where each agent can only see others in the same depot.
    Returns a dictionary mapping agent_id -> list of neighbor agent_ids.
    """
    comm_graph = {}
    for drone_list in depots.values():
        for dn in drone_list:
            # Exclude self from neighbors
            comm_graph[dn] = [other_dn for other_dn in drone_list if other_dn != dn]
    return comm_graph



def delivery_util(reward: float, ie: InteractionEvent) -> float:
    # Compute a realistic utility as expected reward
    p_succ = delivery_success_prob(std_scale=2.0, 
                                   ref_time=ie.timestamps[MODE.FINISH],
                                   ie=ie)
    
    return reward * p_succ


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

    



def delivery_success_prob(std_scale: float, ref_time: float, ie: InteractionEvent) -> float:
    travel_time = ie.timestamps[MODE.RETURN] - ie.timestamps[MODE.FINISH]
    mean = travel_time
    scale = travel_time / std_scale
    x = ie.timestamps[MODE.FINISH] - ref_time 
    prob = epanechnikov_cdf(x, mean, scale)
    return prob


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



def best_response_utility(current_time, ie, reward=1000):
    start, end, avail = ie.time_window
    delivery_est = avail  # or an estimate
    penalty = max(0, delivery_est - end)
    return reward - penalty



def iterative_best_response(server: RoutingAllocation, routing_sim: RoutingSimulator, rng=None, csv_logger=None,
                            init_method = "greedy", trial_id=None, time_step=None):
    max_iters = 20
    depots: Dict[int, List[str]] = {}
    # group drones by depot
    for dn, dp in server.agent_prop_set.items():
        if dp.at_depot:
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
    
    
    # Finding nearby packages
    assigned: Dict[str, str] = {}
    best_ies: Dict[str, Tuple[str, float]] = {}
    final_assignment: Dict[str, Tuple[str, float, InteractionEvent]] = {}
    
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
        

        interaction_events_by_drone: Dict[str, List[InteractionEvent]] = {}
        for dn in drones:
            events = [
                ie for ie in server.agent_prop_set[dn].interaction_events
                if ie.task_name in in_range and
                ie.task_name not in routing_sim.busy_packages
            ]
            interaction_events_by_drone[dn] = events



        if init_method == "random":
            for dn in drones:
                events = interaction_events_by_drone.get(dn, [])
                available_events = [ie for ie in events if ie.task_name not in assigned]
                if available_events:
                    ie = rng.choice(available_events)
                    assigned[dn] = ie.task_name

        elif init_method == "greedy":
            
            for dn in drones:
                best_ie = None
                best_util = float("-inf")
                events = interaction_events_by_drone.get(dn, [])
                for ie in events:

                    if ie.task_name in assigned or ie.task_name in routing_sim.busy_packages:
                        continue
                    # u = compute_utility(
                    #     ie=ie,
                    #     agent=dn,
                    #     assigned=assigned,
                    #     server=server,
                    #     routing_sim=routing_sim,
                    #     mode="exact",  # or "exact"
                    #     lambda_conflict=500,
                    #     delivery_reward=routing_sim.delivery_reward
                    # )
                    u = delivery_util(routing_sim.delivery_reward, ie)
                    if u > best_util:
                        best_util = u
                        best_ie = ie
                if best_ie:
                    assigned[dn] = best_ie.task_name
                    best_ies[dn] = (best_ie.task_name, best_util)
                    final_assignment[dn] = (best_ie.task_name, best_util, best_ie) 
                else:
                    best_ies[dn] = (None, 0.0)

                     

        iter_count = 0
        updated = True
        while updated and iter_count < max_iters:
            updated = False
            for dn in drones:
                if not server.agent_prop_set[dn].at_depot:
                    continue # skip drones in transit
                neighbors = [other_dn for other_dn in drones if other_dn != dn]
                assigned_pkgs = {assigned[n] for n in neighbors if assigned.get(n) is not None}
                best_ie = None
                best_util = float("-inf")
                for ie in interaction_events_by_drone.get(dn, []):
                    pkg = ie.task_name
                    if pkg in assigned_pkgs or pkg in routing_sim.busy_packages or pkg in assigned.values():
                        continue
                    # u = delivery_util(routing_sim.delivery_reward, ie)
                    u = compute_utility(
                        ie=ie,
                        agent=dn,
                        assigned=assigned,
                        server=server,
                        routing_sim=routing_sim,
                        mode="exact",  # or "exact"
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
                    "approx_travel_time": routing_sim.busy_packages[pkg].approx_travel_times.get(dn),
                    "true_travel_time": rt-td
                })