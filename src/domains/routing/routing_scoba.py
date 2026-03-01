import numpy as np
from scipy.stats import gaussian_kde # placeholder for Epanechnikov
from typing import Dict, List, Tuple, Any, Set
from collections import namedtuple, defaultdict
import logging
import math
import random

# Solver types
from solver.scoba_types import DecisionNode, OutcomeNode, SearchTree, InteractionEvent, MODE
from solver.scoba_tree_search import generate_search_tree, get_next_attempt_idx
from solver.scoba_conflict_resolution import SCoBAAlgorithm

from domains.routing.routing_types import EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, travel_time_mean_minutes


TaskUtil = namedtuple("TaskUtil", ["task", "util"])


TRAVEL = dict(
    avg_speed_km_per_min = 0.00777 * 60 / 1.2, # ~0.4662 km/min , just a scale factor to icnrease travel times
    cv = 0.33,           # stdev = cv * mean   (tune 0.2–0.4 to taste)
    dist = "epanechnikov"  # "epanechnikov" or "normal"
)


def delivery_util(reward: float, ie):
    """Constant utility; included for API parity."""
    return reward

def delivery_util_vec(reward: float, ies):
    """Vectorized: returns an array of rewards (one per event)."""
    n = len(ies)
    return np.full(n, reward, dtype=float)

def _epanechnikov_cdf_u(u):
    """CDF of Epanechnikov kernel for standardized u in [-1,1]."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    out[u <= -1] = 0.0
    out[u >= 1]  = 1.0
    mid = (u > -1) & (u < 1)
    um = u[mid]
    out[mid] = 0.5 + 0.75*(um - (um**3)/3.0)
    return out

def travel_time_mean_minutes(loc1, loc2) -> float:
    v1, v2 = convert_to_vector(loc1), convert_to_vector(loc2)
    dist_km = EuclideanLatLongMetric().evaluate(v1, v2)
    mu = dist_km / TRAVEL["avg_speed_km_per_min"]
    return max(math.ceil(mu), 3)

def cdf_travel_time(t_available: float, mu: float) -> float:
    """P(T_out <= t_available) under the configured TRAVEL model."""
    if t_available <= 0:
        return 0.0
    cv = TRAVEL["cv"]
    sigma = max(cv * mu, 1e-6)
    if TRAVEL["dist"].lower() == "epanechnikov":
        u = (t_available - mu) / sigma
        return float(_epanechnikov_cdf_u(u))
    elif TRAVEL["dist"].lower() == "normal":
        # Normal CDF without importing scipy
        z = (t_available - mu) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        raise ValueError(f"Unknown TRAVEL['dist']: {TRAVEL['dist']}")


def delivery_success_prob_common(
    depot_loc,
    pkg_loc,
    ref_time: float,
    ie
) -> float:
    """
    Probability of delivering within the package's latest time,
    if the drone attempts at max(ref_time, window start).
    """
    start_t  = ie.timestamps[MODE.START]
    finish_t = ie.timestamps[MODE.FINISH]
    attempt_t = max(ref_time, start_t)          # can't start before window OR before you're free
    slack = finish_t - attempt_t                # time available to fly outbound
    mu_out = travel_time_mean_minutes(depot_loc, pkg_loc)
    return cdf_travel_time(slack, mu_out)


def make_success_prob_fn(sim, server, drone_nm):
    depot_loc = server.agent_set[drone_nm].depot_loc

    def success_prob(ref_time, ie):
        pkg_nm = getattr(ie, "task_name", None)
        # Prefer active pkgs (tree is built over actives)
        if pkg_nm in sim.active_packages:
            pkg_loc = sim.active_packages[pkg_nm].delivery
        elif pkg_nm in sim.busy_packages:
            pkg_loc = sim.busy_packages[pkg_nm].delivery
        else:
            # Fallback: look up from server windows if needed
            # (should rarely happen in tree generation)
            return 0.0
        return delivery_success_prob_common(depot_loc, pkg_loc, ref_time, ie)

    return success_prob


def scoba_routing(server, routing_sim, rng: Any = None, csv_logger=None, trial_id=None, time_step=None, 
                  comms_dict=None, allow_overlap=False, depot_order=None) -> None:
    """
    Assign tasks to drones using the SC0BA conflict-based allocation algorithm.
    `server` is a RoutingAllocation, `routing_sim` a RoutingSimulator.
    """
    if not allow_overlap: 
        # 1) Group available drones by depot
        # logging.info(f"[t={server.current_time}] Starting SCoBA assignment...")
        available_drones = {}  # depot_number -> [drone_names]
        for drone_nm, dp in server.agent_prop_set.items():
            drone = server.agent_set[drone_nm]
            if dp.at_depot is True:
                available_drones.setdefault(drone.depot_number, []).append(drone_nm)

        # Diagnostics to send to CBA
        task_util_allocation = {}       # drone_nm -> TaskUtil(task=<pkg_nm>, util=<value>)
        all_considered_tasks = {}       # drone_nm -> set(pkg_nms)
        assignment_util = 0.0


        def util_val_fn(ie):
            return routing_sim.delivery_reward
        
        # a factory the coordinator can call: given drone -> returns (ref_time, ie) -> prob
        def success_prob_factory(drone_nm: str):
            return make_success_prob_fn(routing_sim, server, drone_nm)



        # 2) Assign all available drones, grouped by depot
        for depot_number, depot_drones in available_drones.items():
            # Any drone from this depot has the same depot_loc
            depot_loc = server.agent_set[depot_drones[0]].depot_loc


            # Find packages in range of this depot
            pkgs_in_range = set()
            for pkg_nm, pp in routing_sim.active_packages.items():
                dist = EuclideanLatLongMetric().evaluate(
                    convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
                )
                if dist <= routing_sim.distance_thresh:
                    pkgs_in_range.add(pkg_nm)


            depot_assigned_pkgs = set()
            # Priority ordering among drones from the same depot
            for drone_nm in depot_drones:
                # Exclude already tentatively assigned packages (from this depot)
                pkgs_to_consider = pkgs_in_range - depot_assigned_pkgs
            
                sp_fn = success_prob_factory(drone_nm)
                # Generate the single-agent search tree
                # Build the single-agent tree
                generate_search_tree(
                    server,
                    drone_nm,
                    pkgs_to_consider,
                    sp_fn,              # <-- success prob callable: (ref_time, ie) -> [0,1]
                    util_val_fn,
                    server.current_time 
                )
                tree = server.agent_prop_set[drone_nm].tree

                if tree:  # not empty
                    dec_idx = get_next_attempt_idx(tree)
                    if dec_idx != -1:
                        dec_node = tree.nodes[dec_idx]
                        # Tentatively mark this package as taken by a drone from this depot
                        depot_assigned_pkgs.add(dec_node.task_name)

                        # Record diagnostics for coordination step
                        assignment_util += dec_node.util
                        task_util_allocation[drone_nm] = TaskUtil(task=dec_node.task_name, util=dec_node.util)
                        all_considered_tasks[drone_nm] = set(pkgs_to_consider)

        # 3) Resolve conflicts across depots via SCoBA coordination    
        # 
           
        if task_util_allocation:
            scoba_alg = SCoBAAlgorithm(allocation=server, routing_sim=routing_sim)
            if comms_dict is None:
                # scoba_alg = SCoBAAlgorithm(allocation=server, routing_sim=routing_sim)
                true_task_util_allocation = scoba_alg.coordinate_allocation(
                    task_util_allocation,
                    all_considered_tasks,
                    assignment_util,
                    success_prob_factory,
                    util_val_fn,
                    0.0,
                )
            else:
                # print("using dec")
                # --- Build communication neighborhoods between depots ---
                depots_to_drones: Dict[int, List[str]] = {d: sorted(dr_list) for d, dr_list in available_drones.items()}
                all_depots: Set[int] = set(depots_to_drones.keys()) | set(comms_dict.keys())
                depot_neighbors = {d: set([d]) | set(comms_dict.get(d, [])) for d in all_depots}

                # Also guard against depots present in comms_dict with no drones right now
                for d in comms_dict.keys():
                    depot_neighbors.setdefault(d, set([d]) | set(comms_dict.get(d, [])))
                
                
                # Map depot -> all drones in its *visible* neighborhood (incl. self depot)
                visible_drones_by_depot: Dict[int, List[str]] = {}
                for d in depot_neighbors:
                    vis = []
                    for nd in depot_neighbors[d]:
                        vis.extend(depots_to_drones.get(nd, []))
                    visible_drones_by_depot[d] = vis

                for d in sorted(visible_drones_by_depot.keys()):
                    group_drones = [dn for dn in visible_drones_by_depot[d]
                                    if dn in task_util_allocation and dn not in server.agent_task_allocation]

                    if not group_drones:
                        continue
                
                    # Filter the group's proposed tasks to those still active (not already committed by prior groups)
                    # and assemble per-drone considered sets (dict, not a set-of-sets).
                    task_util_in_group: Dict[str, TaskUtil] = {}
                    considered_tasks_in_group: Dict[str, Set[str]] = {}
                    
                    for dn in group_drones:
                        # proposed task for dn
                        tu = task_util_allocation.get(dn)
                        if not tu:
                            continue
                        if tu.task not in routing_sim.active_packages:
                            # proposed package already taken by earlier group ⇒ still include dn but their proposed
                            # may be invalid; SCoBA will see it as unavailable if not in considered set.
                            pass
                        task_util_in_group[dn] = tu

                        # only keep still-active tasks in the "considered" set to prevent SCoBA from picking removed pkgs
                        considered = {pkg for pkg in all_considered_tasks.get(dn, set())
                                    if pkg in routing_sim.active_packages}
                        # Edge case: if empty, still include empty set; SCoBA may then skip dn
                        considered_tasks_in_group[dn] = considered

                    if not task_util_in_group:
                        continue

                    assignment_util_in_group = sum(tu.util for tu in task_util_in_group.values())

                    # Run SCoBA just for this depot's visible neighborhood
                    true_task_util_allocation = scoba_alg.coordinate_allocation(
                        task_util_in_group,
                        considered_tasks_in_group,  # Dict[str, Set[str]]
                        assignment_util_in_group,
                        success_prob_factory,
                        util_val_fn,
                        0.0,
                    )            

            # there might be redundancies in true_task_util_allocation, ignore them.
            for drone_nm, pkg_util in true_task_util_allocation.items():
                # Defensive access whether it's a namedtuple/object/dict
                pkg_nm = pkg_util.task
                depot_loc = server.agent_set[drone_nm].depot_loc
                # Ensure drone isn't already assigned
                assert drone_nm not in server.agent_task_allocation

                if pkg_nm not in routing_sim.busy_packages:
                    # Assign (drone -> package) with time = Inf (unknown attempt time placeholder)
                    server.agent_task_allocation[drone_nm] =  (pkg_nm, float("inf"))


                    window = routing_sim.active_packages[pkg_nm].time_window
                    delivery_location = routing_sim.active_packages[pkg_nm].delivery
                    td, rt = sample_true_delivery_return_time(
                            depot_loc,
                            delivery_location,
                            window,
                            server.current_time,
                            rng,
                        )
                    routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = (td, rt)

                    # Update drone state
                    server.agent_prop_set[drone_nm].at_depot = False
                    server.agent_prop_set[drone_nm].current_package = pkg_nm

                    # Move package from active -> busy
                    new_busy_package = routing_sim.active_packages[pkg_nm]
                    routing_sim.busy_packages[pkg_nm] = new_busy_package
                    routing_sim.active_packages.pop(pkg_nm, None)
                    routing_sim.num_active_packages -= 1
                    depot_number = server.agent_set[drone_nm].depot_number

                    if csv_logger:
                        csv_logger.log("drone_assignment.csv",
                        {
                            "trial": trial_id,
                            "time": time_step,
                            "drone_id": drone_nm,
                            "depot_number": depot_number,
                            "pkg_id": pkg_nm,
                            "pkg_earliest_time": server.agent_task_windows[(drone_nm, pkg_nm)][0],
                            "pkg_latest_time": server.agent_task_windows[(drone_nm, pkg_nm)][1],
                            "agent_tw_avail": server.agent_task_windows[(drone_nm, pkg_nm)][2],
                            "true_return_time": rt,
                            "true_delivery_time": td,
                            "approx_travel_time": routing_sim.busy_packages[pkg_nm].approx_travel_times.get(depot_number),
                            "true_travel_time": rt-td
                            }
                        )   

    else:
        # 1) Group available drones by depot
        # logging.info(f"[t={server.current_time}] Starting SCoBA assignment...")
        available_drones = {}  # depot_number -> [drone_names]
        depots: Dict[int, List[str]] = {}
        global_depos: Dict[int, List[str]] = {} 
        # group drones by depot
        for dn, dp in server.agent_prop_set.items():
            dnum = server.agent_set[dn].depot_number
            if dp.at_depot:
                depots.setdefault(dnum, []).append(dn)
            global_depos.setdefault(dnum, []).append(dn)

        if not any(depots.values()):
            logging.info(f"[Scoba] t={time_step} no drones at depot; skipping assignment.")
            return        
        
        # Diagnostics to send to CBA
        task_util_allocation = {}       # drone_nm -> TaskUtil(task=<pkg_nm>, util=<value>)
        all_considered_tasks = {}       # drone_nm -> set(pkg_nms)
        assignment_util = 0.0

        # Depot info considering communication neighborhoods between depots 
        all_depots_comm_net = set(depots.keys())
        global_depots_comm_net = set(global_depos.keys()) 
        if comms_dict is None:
            # fully connected: each depot sees all depots (including itself)
            depot_neighbors = {d: set(all_depots_comm_net) for d in all_depots_comm_net}
            global_depots_neighbors = {d: set(global_depots_comm_net) for d in global_depots_comm_net}
        else:
            depot_neighbors = {d: set([d]) | set(comms_dict.get(d, [])) for d in all_depots_comm_net }
            global_depots_neighbors = {d: set([d]) | set(comms_dict.get(d, [])) for d in global_depots_comm_net}
            # Also guard against depots present in comms_dict with no drones right now
            for d in comms_dict.keys():
                depot_neighbors.setdefault(d, set([d]) | set(comms_dict.get(d, [])))
                global_depots_neighbors.setdefault(d, set([d]) | set(comms_dict.get(d, [])))
        
        
        visible_by_drone = {}
        for depot_num, drones in depots.items():
            vis_drones = []
            for nd in depot_neighbors.get(depot_num, {depot_num}):
                vis_drones.extend(depots.get(nd, []))
            vis_set = set(vis_drones)
            for dn in drones:
                visible_by_drone[dn] = sorted(x for x in vis_set if x != dn)


        # Map depot -> all drones in its *visible* neighborhood (incl. self depot)
        visible_drones_by_depot: Dict[int, List[str]] = {}
        global_visible_drones_by_depot: Dict[int, List[str]] = {}

        for d in depot_neighbors:
            vis = []
            for nd in depot_neighbors[d]:
                vis.extend(depots.get(nd, []))
            visible_drones_by_depot[d] = vis

        for d in global_depots_neighbors:
            vis = []
            for nd in global_depots_neighbors[d]:
                vis.extend(global_depos.get(nd, []))
            global_visible_drones_by_depot[d] = vis


        previously_pkgs_visible_to_depot = {d: set() for d in global_depos.keys()}
        active_now = set(routing_sim.active_packages.keys())
    
        for depot_num, drones in global_depos.items():
            visible_drones = set(global_visible_drones_by_depot.get(depot_num, []))  # comms neighborhood drones
            prev = set()
            for pkg, claimants in getattr(routing_sim, "package_claims", {}).items():
                if pkg not in active_now:
                    continue
                claimants = set(claimants) if not isinstance(claimants, set) else claimants
                if claimants & visible_drones:
                    prev.add(pkg)
            
            # 2) packages with a winner that is visible, but only if still active
            # usually short-lived if you pop active on delivery; safe anyway
            for pkg, winner_dn in getattr(routing_sim, "package_winners", {}).items():
                if pkg not in active_now:
                    continue
                if winner_dn in visible_drones:
                    prev.add(pkg)

            previously_pkgs_visible_to_depot[depot_num] = prev

        attempting_visible_to_depot: Dict[int, Dict[str, List[str]]] = {d: {} for d in global_depos.keys()}

        def util_val_fn(ie):
            return routing_sim.delivery_reward
        
        # a factory the coordinator can call: given drone -> returns (ref_time, ie) -> prob
        def success_prob_factory(drone_nm: str):
            return make_success_prob_fn(routing_sim, server, drone_nm)

        if routing_sim.package_claims:
            for depot_num in global_depos.keys():
                visible_set = set(global_visible_drones_by_depot[depot_num])

                who = {}
                for pkg, claimants in routing_sim.package_claims.items():
                    if not isinstance(claimants, set):
                                        claimants = set(claimants)
                    vis_claimants = sorted(claimants & visible_set)
                    if vis_claimants:
                        who[pkg] = vis_claimants

                attempting_visible_to_depot[depot_num] = who

        true_task_util_allocation = {}   # FINAL merged allocation
        assigned_pkgs = set()            # packages already committed by previous groups
        assigned_drones = set()          # drones already committed by previous groups


        for d in sorted(attempting_visible_to_depot.keys()):
            logging.info(f"[Scoba] depot {d} visible-attempts: {attempting_visible_to_depot[d]}")


        for d in sorted(previously_pkgs_visible_to_depot.keys()):
            logging.info(
                f"[Scoba] depot {d} previously-visible pkgs: {sorted(previously_pkgs_visible_to_depot[d])}"
            )                                                               

            # --- Precompute interaction events in range per depot ---
        interaction_events_by_drone: Dict[str, List[InteractionEvent]] = {}
        in_range_by_depot: Dict[int, set] = {}

        if depot_order == "asc":
            depot_order = sorted(depots.keys())
        elif depot_order == "desc": 
            depot_order = sorted(depots.keys(), reverse=True)
        elif depot_order == "random":
            depot_order = random.sample(list(depots.keys()), len(depots))

        # 2) Assign all available drones, grouped by depot
        for depot_number in depot_order:
            # Any drone from this depot has the same depot_loc
            depot_drones = depots[depot_number]
            depot_loc = server.agent_set[depot_drones[0]].depot_loc


            # Find packages in range of this depot
            pkgs_in_range = set()
            for pkg_nm, pp in routing_sim.active_packages.items():
                dist = EuclideanLatLongMetric().evaluate(
                    convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
                )
                if dist <= routing_sim.distance_thresh:
                    pkgs_in_range.add(pkg_nm)
            in_range_by_depot[depot_number] = pkgs_in_range

            for dn in depot_drones:

                events = [
                    ie for ie in server.agent_prop_set[dn].interaction_events
                    if ie.task_name in pkgs_in_range and ie.task_name not in previously_pkgs_visible_to_depot[depot_number]
                ]
                interaction_events_by_drone[dn] = events
            depot_assigned_pkgs = set()
            # Priority ordering among drones from the same depot
            for drone_nm in depot_drones:
                # Exclude already tentatively assigned packages (from this depot)
                pkgs_to_consider = in_range_by_depot[depot_number] - depot_assigned_pkgs - previously_pkgs_visible_to_depot[depot_number]
            
                sp_fn = success_prob_factory(drone_nm)
                # Generate the single-agent search tree
                # Build the single-agent tree
                generate_search_tree(
                    server,
                    drone_nm,
                    pkgs_to_consider,
                    sp_fn,              # <-- success prob callable: (ref_time, ie) -> [0,1]
                    util_val_fn,
                    server.current_time 
                )
                tree = server.agent_prop_set[drone_nm].tree

                if tree:  # not empty
                    dec_idx = get_next_attempt_idx(tree)
                    if dec_idx != -1:
                        dec_node = tree.nodes[dec_idx]
                        # Tentatively mark this package as taken by a drone from this depot
                        depot_assigned_pkgs.add(dec_node.task_name)

                        # Record diagnostics for coordination step
                        assignment_util += dec_node.util
                        task_util_allocation[drone_nm] = TaskUtil(task=dec_node.task_name, util=dec_node.util)
                        all_considered_tasks[drone_nm] = set(pkgs_to_consider)

        # 3) Resolve conflicts across depots via SCoBA coordination    
        if task_util_allocation:
            scoba_alg = SCoBAAlgorithm(allocation=server, routing_sim=routing_sim)
            no_comms = all(len(neigh) == 0 for neigh in comms_dict.values())

            if no_comms:
                # No inter-depot communication -> skip SCoBA
                true_task_util_allocation = task_util_allocation
            else:
                # print("using dec")
                # --- Build communication neighborhoods between depots ---
                depots_to_drones: Dict[int, List[str]] = {d: sorted(dr_list) for d, dr_list in depots.items()}
                all_depots: Set[int] = set(depots_to_drones.keys()) | set(comms_dict.keys())
                depot_neighbors = {d: set([d]) | set(comms_dict.get(d, [])) for d in all_depots}

                # Also guard against depots present in comms_dict with no drones right now
                for d in comms_dict.keys():
                    depot_neighbors.setdefault(d, set([d]) | set(comms_dict.get(d, [])))
                
                
                # Map depot -> all drones in its *visible* neighborhood (incl. self depot)
                visible_drones_by_depot: Dict[int, List[str]] = {}
                for d in depot_neighbors:
                    vis = []
                    for nd in depot_neighbors[d]:
                        vis.extend(depots_to_drones.get(nd, []))
                    visible_drones_by_depot[d] = vis
                true_task_util_allocation = {}
                for d in sorted(visible_drones_by_depot.keys()):
                    group_drones = [
                        dn for dn in visible_drones_by_depot[d]
                        if dn in task_util_allocation and dn not in server.agent_task_allocation
                    ]
                    if not group_drones:
                        continue
                
                    # Filter the group's proposed tasks to those still active (not already committed by prior groups)
                    # and assemble per-drone considered sets (dict, not a set-of-sets).
                    task_util_in_group: Dict[str, TaskUtil] = {}
                    considered_tasks_in_group: Dict[str, Set[str]] = {}
                    
                    for dn in group_drones:
                        # proposed task for dn
                        tu = task_util_allocation.get(dn)
                        if not tu:
                            continue
                        if tu.task not in routing_sim.active_packages:
                            # proposed package already taken by earlier group ⇒ still include dn but their proposed
                            # may be invalid; SCoBA will see it as unavailable if not in considered set.
                            pass
                        task_util_in_group[dn] = tu

                        # only keep still-active tasks in the "considered" set to prevent SCoBA from picking removed pkgs
                        considered = {pkg for pkg in all_considered_tasks.get(dn, set())
                                    if pkg in routing_sim.active_packages}
                        # Edge case: if empty, still include empty set; SCoBA may then skip dn
                        considered_tasks_in_group[dn] = considered

                    if not task_util_in_group:
                        continue

                    assignment_util_in_group = sum(tu.util for tu in task_util_in_group.values())

                    # Run SCoBA just for this depot's visible neighborhood
                    group_allocation = scoba_alg.coordinate_allocation(
                        task_util_in_group,
                        considered_tasks_in_group,
                        assignment_util_in_group,
                        success_prob_factory,
                        util_val_fn,
                        0.0,
                    )        


                    for dn, tu in group_allocation.items():
                        true_task_util_allocation[dn] = tu

            # there might be redundancies in true_task_util_allocation, ignore them.
            for drone_nm, pkg_util in true_task_util_allocation.items():
                # Defensive access whether it's a namedtuple/object/dict
                pkg_nm = pkg_util.task
                depot_loc = server.agent_set[drone_nm].depot_loc

                if pkg_nm not in previously_pkgs_visible_to_depot[server.agent_set[drone_nm].depot_number]:
                    # Assign (drone -> package) with time = Inf (unknown attempt time placeholder)
                    server.agent_task_allocation[drone_nm] =  (pkg_nm, float("inf"))


                    window = routing_sim.active_packages[pkg_nm].time_window
                    delivery_location = routing_sim.active_packages[pkg_nm].delivery
                    td, rt = sample_true_delivery_return_time(
                            depot_loc,
                            delivery_location,
                            window,
                            server.current_time,
                            rng,
                        )
                    routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = (td, rt)

                    # Update drone state
                    server.agent_prop_set[drone_nm].at_depot = False
                    server.agent_prop_set[drone_nm].current_package = pkg_nm

                    # Move package from active -> busy
                    routing_sim.package_claims.setdefault(pkg_nm, set()).add(drone_nm)
                    routing_sim.package_registry[pkg_nm]["claimed_by"].append(drone_nm)
                    routing_sim.package_registry[pkg_nm]["time_assigned"].append(time_step)
                    routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages.get(pkg_nm)
                    # new_busy_package = routing_sim.active_packages[pkg_nm]
                    # routing_sim.busy_packages[pkg_nm] = new_busy_package
                    # routing_sim.active_packages.pop(pkg_nm, None)
                    # routing_sim.num_active_packages -= 1
                    depot_number = server.agent_set[drone_nm].depot_number

                    if csv_logger:
                        csv_logger.log("drone_assignment.csv",
                        {
                            "trial": trial_id,
                            "time": time_step,
                            "drone_id": drone_nm,
                            "depot_number": depot_number,
                            "pkg_id": pkg_nm,
                            "pkg_earliest_time": server.agent_task_windows[(drone_nm, pkg_nm)][0],
                            "pkg_latest_time": server.agent_task_windows[(drone_nm, pkg_nm)][1],
                            "agent_tw_avail": server.agent_task_windows[(drone_nm, pkg_nm)][2],
                            "true_return_time": rt,
                            "true_delivery_time": td,
                            "approx_travel_time": routing_sim.busy_packages[pkg_nm].approx_travel_times.get(depot_number),
                            "true_travel_time": rt-td
                            }
                        )   
                