import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import deque, Counter
import logging
import heapq
import random

# from domains.routing.routing_scoba import delivery_success_prob_common, cdf_travel_time
from solver.scoba_types import InteractionEvent, MODE, GenericAllocation as RoutingAllocation
from domains.routing.routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, travel_time_mean_minutes
from .travel_model import delivery_success_prob_ibr

random.seed(42)


# def create_comm_graph(depots: Dict[int, List[str]]) -> Dict[str, List[str]]:
#     """
#     Create a communication graph where each agent can only see others in the same depot.
#     Returns a dictionary mapping agent_id -> list of neighbor agent_ids.
#     """
#     comm_graph = {}
#     for drone_list in depots.values():
#         for dn in drone_list:
#             # Exclude self from neighbors
#             comm_graph[dn] = [other_dn for other_dn in drone_list if other_dn != dn]
#     return comm_graph



# def group_welfare(assignments, group, p_cache, reward):
#     # assignments: dict drone->pkg (pkg can be None)
#     # group: iterable of drones to include
#     # returns sum_t reward * max_{drone in group assigned to t} p(drone,t)
#     best_by_pkg = {}
#     for dn in group:
#         pkg = assignments.get(dn)
#         if pkg is None:
#             continue
#         p = p_cache.get((dn, pkg), 0.0)
#         if (pkg not in best_by_pkg) or (p > best_by_pkg[pkg]):
#             best_by_pkg[pkg] = p
#     return reward * sum(best_by_pkg.values())


def group_welfare(assignments, group, p_cache, reward):
    # W(x) = sum_t v_t * (1 - prod_{i assigned to t} (1 - p_i))   [Eq. (3)]
    fail_by_pkg = {}
    for dn in group:
        pkg = assignments.get(dn)
        if pkg is None:
            continue
        p = p_cache.get((dn, pkg), 0.0)
        fail_by_pkg[pkg] = fail_by_pkg.get(pkg, 1.0) * (1.0 - p)


    return reward * sum(1.0 - f for f in fail_by_pkg.values())


# def group_welfare_old(assignments, group, p_cache, reward):
#     best_by_pkg = {}
#     for dn in group:
#         pkg = assignments.get(dn)
#         if pkg is None:
#             continue
#         p = p_cache.get((dn, pkg), 0.0)
#         if (pkg not in best_by_pkg) or (p > best_by_pkg[pkg]):
#             best_by_pkg[pkg] = p
#         # print(f"Drone {dn} assigned to pkg {pkg} with p={p:.2f}")
    
#     return reward * sum(best_by_pkg.values())

def compute_utility(pkg, agent, assigned, group, p_cache, reward):
    base = dict(assigned); base[agent] = None
    cand = dict(assigned); cand[agent] = pkg
    w0, w1 = group_welfare(base, group, p_cache, reward), group_welfare(cand, group, p_cache, reward)
    # others = [d for d in group if d != agent and assigned.get(d) == pkg]
    # if others:
    #     o0 = group_welfare_old(base, group, p_cache, reward)
    #     o1 = group_welfare_old(cand, group, p_cache, reward)
    #     print(f"[Wi] a={agent} k={pkg} p_i={p_cache.get((agent,pkg),0.0):.3f} "
    #           f"p_incumbent={[round(p_cache.get((d,pkg),0.0),3) for d in others]} || "
    #           f"Eq3: W0={w0:.1f} W1={w1:.1f} U={w1-w0:.1f} || "
    #           f"max: W0={o0:.1f} W1={o1:.1f} U={o1-o0:.1f}", flush=True)
    return w1 - w0

# DUP_HITS = Counter()

# def compute_utility(pkg, agent, assigned, group, p_cache, reward):
#     base = dict(assigned); base[agent] = None
#     cand = dict(assigned); cand[agent] = pkg
#     w0 = group_welfare(base, group, p_cache, reward)
#     w1 = group_welfare(cand, group, p_cache, reward)

#     others = [d for d in group if d != agent and assigned.get(d) == pkg]
#     if others:
#         u_old = (group_welfare_old(cand, group, p_cache, reward)
#                  - group_welfare_old(base, group, p_cache, reward))
#         DUP_HITS["dup"] += 1
#         print(f"[Wi] a={agent} k={pkg} p_i={p_cache.get((agent,pkg),0.0):.3f} "
#               f"G_k={[(d, round(p_cache.get((d,pkg),0.0),3)) for d in others]} "
#               f"Wi_idle={w0:.2f} Wi_k={w1:.2f} U_new={w1-w0:.2f} U_old={u_old:.2f}",
#               flush=True)
#     return w1 - w0


def iterative_best_response(server: RoutingAllocation, routing_sim: RoutingSimulator, rng=None, csv_logger=None,
                            init_method = "empty", trial_id=None, time_step=None, comms_dict=None, allow_overlap=False, 
                            depot_order="asc"):





    # might be redundant but just to separate those at depot without communication constraints
    depots: Dict[int, List[str]] = {}
    global_depos: Dict[int, List[str]] = {} # depots and drones dictionary 
    # group drones by depot
    for dn, dp in server.agent_prop_set.items():
        dnum = server.agent_set[dn].depot_number
        if dp.at_depot:
            depots.setdefault(dnum, []).append(dn)
        global_depos.setdefault(dnum, []).append(dn)
    # logging.info(f"Drones at depot information: {depots}")

    if not any(depots.values()):
        logging.info(f"[IBR] t={time_step} no drones at depot; skipping assignment.")
        return


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


    logging.info(
        f"[IBR] t={time_step} now={routing_sim.current_time} "
        f"active={len(routing_sim.active_packages)} busy={len(routing_sim.busy_packages)} "
        f"claims={len(getattr(routing_sim,'package_claims',{}))} winners={len(getattr(routing_sim,'package_winners',{}))}"
    )



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

    for d in sorted(attempting_visible_to_depot.keys()):
        logging.info(f"[IBR] depot {d} visible-attempts: {attempting_visible_to_depot[d]}")


    for d in sorted(previously_pkgs_visible_to_depot.keys()):
        logging.info(
            f"[IBR] depot {d} previously-visible pkgs: {sorted(previously_pkgs_visible_to_depot[d])}"
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

    for depot_num in depot_order:
        drones = depots[depot_num]
        depot_loc = server.agent_set[drones[0]].depot_loc
        
        in_range = set()
        for pkg, pp in routing_sim.active_packages.items():
            dist = EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
            )
            if dist <= routing_sim.distance_thresh:
                in_range.add(pkg)
            
        in_range_by_depot[depot_num] = in_range

        if allow_overlap:
            # get the winners from previous rounds

            for dn in drones:

                events = [
                    ie for ie in server.agent_prop_set[dn].interaction_events
                    if ie.task_name in in_range and ie.task_name not in previously_pkgs_visible_to_depot[depot_num]
                ]
                interaction_events_by_drone[dn] = events
        else:
            for dn in drones:
                events = [
                    ie for ie in server.agent_prop_set[dn].interaction_events
                    if ie.task_name in in_range and ie.task_name not in routing_sim.busy_packages
                    and ie.task_name not in routing_sim.package_claims
                ]
                interaction_events_by_drone[dn] = events
        
    p_cache = {}  
    for dn, events in interaction_events_by_drone.items():
        for ie in events:
            p_cache[(dn, ie.task_name)] = delivery_success_prob_ibr(
                ref_time=routing_sim.current_time,
                ie=ie,
                std_scale=routing_sim.tt_est_std_scale,
            )
    

    # --- Initial assignment (per depot)
    assigned: Dict[str, Optional[str]] = {}
    # best_ies: Dict[str, Tuple[str, float]] = {}
    final_assignment: Dict[str, Tuple[str, float, InteractionEvent]] = {}

    for depot_num in depot_order:
        drones = depots[depot_num]
        assigned_per_depot = []
        
        if init_method == "empty":
            for dn in drones:
                assigned[dn] = None
        elif init_method == "random":
            for dn in drones:
                events = interaction_events_by_drone.get(dn, [])   
            if len(events) == 0:
                # best_ies[dn] = (None, 0.0)
                continue         
            if allow_overlap is False:
                available_events = [ie for ie in events if ie.task_name not in assigned_per_depot and ie.task_name not in routing_sim.busy_packages]
            else:
                available_events = [ie for ie in events if ie.task_name not in assigned_per_depot and ie.task_name not in previously_pkgs_visible_to_depot[depot_num]]

            # safeguard
            if len(available_events) == 0:
                continue

            ie = rng.choice(available_events)
            assigned[dn] = ie.task_name
            assigned_per_depot.append(ie.task_name)

            final_assignment[dn] = (ie.task_name, 0.0, ie)
        elif init_method == "greedy":
            for dn in drones:
                best_ie, best_util = None, float("-inf")
                for ie in interaction_events_by_drone.get(dn, []):
                    pkg = ie.task_name
                    if not allow_overlap:
                        if pkg in assigned_per_depot or pkg in routing_sim.busy_packages:
                            continue
                    else:
                        if pkg in assigned_per_depot or pkg in previously_pkgs_visible_to_depot[depot_num]:
                            continue
                    u = routing_sim.delivery_reward * p_cache.get((dn, pkg), 0.0)

                    if u > best_util:
                        best_util, best_ie = u, ie

                if best_ie is not None:
                    assigned[dn] = best_ie.task_name
                    assigned_per_depot.append(best_ie.task_name)
                    final_assignment[dn] = (best_ie.task_name, best_util, best_ie)
                else:
                    assigned[dn] = None


        logging.info(f"[IBR] initial assigned: { {dn: assigned.get(dn) for dn in sorted(assigned)} }")      
    # --- Iterative best response (GLOBAL), information-aware ---
    # Important: each drone "sees" only drones from depots in its comms neighborhood.
    k_rounds = 500
    # Ensure iteration over depots.values() follows depot_order and is deterministic
    depots = {d: sorted(depots[d]) for d in depot_order if d in depots}
    all_considered_drones = [dn for drones in depots.values() for dn in drones]
    rng.shuffle(all_considered_drones)

    rounds_completed = 0

    for r in range(1, k_rounds + 1):
        changes_this_round = 0
        rng.shuffle(all_considered_drones)
        # print(f"Iterations {r}, {all_considered_drones}")
        for drone in all_considered_drones:

            if not server.agent_prop_set[drone].at_depot:
                continue

            visible_drones = visible_by_drone.get(drone, [])
            assigned_visible = {
                n: assigned[n]
                for n in visible_drones
                if assigned.get(n) is not None
            }

            dnum = server.agent_set[drone].depot_number
            prev_blocked = previously_pkgs_visible_to_depot.get(dnum, set())

            if allow_overlap:
                blocked_pkgs = set(prev_blocked)
            else:
                blocked_pkgs = (
                    set(assigned_visible.values())
                    | set(prev_blocked)
                    | set(routing_sim.busy_packages)
                )

            best_ie = None
            best_util = 0.0  # idle utility

            for ie in interaction_events_by_drone.get(drone, []):
                pkg = ie.task_name
                if allow_overlap:
                    if pkg in prev_blocked:  # only block cross-timestep committed pkgs
                        continue
                else:
                    if pkg in blocked_pkgs:  # block busy + neighbor assignments + prev_blocked
                        continue
                # enforce blocking if desired
                # if pkg in blocked_pkgs:
                #     continue

                group = [drone] + visible_by_drone.get(drone, [])

                # IMPORTANT: use the CURRENT assignments for the group, not assigned_visible
                assigned_group = {dn: assigned.get(dn) for dn in group}

                u = compute_utility(
                    pkg=pkg,
                    agent=drone,
                    assigned=assigned_group,
                    group=group,
                    p_cache=p_cache,
                    reward=routing_sim.delivery_reward
                )


                if u > best_util:
                    best_util = u
                    best_ie = ie

            current_pkg = assigned.get(drone)
            new_pkg = best_ie.task_name if best_ie is not None else None

            if new_pkg != current_pkg:
                assigned[drone] = new_pkg
                if best_ie is not None:
                    final_assignment[drone] = (new_pkg, best_util, best_ie)
                changes_this_round += 1
            # print(f"Time step {time_step} Round {r}, Drone {drone}: current pkg={current_pkg}, new pkg={new_pkg}, best util={best_util:.2f}")

        rounds_completed += 1



        if changes_this_round == 0:
            break
        
    # ## DELETE
    # n_cands = len({ie.task_name for dn in all_considered_drones
    #             for ie in interaction_events_by_drone.get(dn, [])})
    # pvals = [v for v in p_cache.values() if v > 0]
    # print(f"[SLACK] t={time_step} at_depot={len(all_considered_drones)} cands={n_cands} "
    #       f"rounds={rounds_completed} dup_evals={DUP_HITS['dup']} "
    #       f"idle={sum(1 for dn in all_considered_drones if assigned.get(dn) is None)} "
    #       f"p_pos={len(pvals)}/{len(p_cache)} p_max={max(pvals, default=0.0):.3f}",
    #       flush=True)
    # DUP_HITS.clear()
    

    csv_logger.log("computational_efficiency_metrics.csv", {
        "trial": trial_id,
        "time": time_step,

        "k_rounds": rounds_completed,


    })
    
    logging.info("[IBR] Final assignments are: " + str(assigned))

    for dn, pkg in assigned.items():
        if pkg is None:
            continue    
        if allow_overlap is False:
            if pkg in routing_sim.busy_packages:
                logging.warning(f"Drone {dn} lost allocation for {pkg} (already taken)")
                continue

            if pkg not in routing_sim.active_packages:   # ← add this
                logging.warning(f"Drone {dn} lost allocation for {pkg} (already popped)")
                continue
            else:
                server.agent_task_allocation[dn] = (pkg, float('inf'))
                window = routing_sim.active_packages[pkg].time_window
                depot_loc = server.agent_set[dn].depot_loc
                delivery_location = routing_sim.active_packages[pkg].delivery
                td, rt = sample_true_delivery_return_time(
                    depot_loc,
                    delivery_location,
                    window,
                    server.current_time,
                    rng,
                )
                routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
                server.agent_prop_set[dn].at_depot = False
                server.agent_prop_set[dn].current_package = pkg

                routing_sim.busy_packages[pkg] = routing_sim.active_packages.pop(pkg)
                routing_sim.num_active_packages -= 1
                final_pkg, util, ie = final_assignment[dn]
                assert final_pkg == pkg, f"Final assigned pkg mismatch for {dn}: expected {pkg}, got {final_pkg}"
                # assert len(assigned.values()) == len(set(assigned.values())), "Duplicate package assignments found!"

                if csv_logger:
                    depot_number = server.agent_set[dn].depot_number
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
                        "approx_travel_time": routing_sim.busy_packages[pkg].approx_travel_times.get(depot_number),
                        "true_travel_time": rt-td
                    })
        else:
            # if pkg in routing_sim.package_winners:
            #     logging.warning(f"Drone {dn} skipped: {pkg} already has a winner.")
            #     continue

            # Record this drone’s claim (allow multiple claimants globally)
            if pkg is None:
                continue
            routing_sim.package_claims.setdefault(pkg, set()).add(dn)
            routing_sim.package_registry[pkg]["claimed_by"].append(dn)
            routing_sim.package_registry[pkg]["time_assigned"].append(time_step)
            routing_sim.busy_packages[pkg] = routing_sim.active_packages.get(pkg)

            # Keep package in active until someone actually delivers.
            # DO NOT move active -> busy here; the race resolves later at arrival.

            # Schedule this drone’s true (delivery, return) now (each claimant has its own trajectory)
            window = routing_sim.active_packages[pkg].time_window
            depot_loc = server.agent_set[dn].depot_loc
            delivery_location = routing_sim.active_packages[pkg].delivery
            td, rt = sample_true_delivery_return_time(
                depot_loc,
                delivery_location,
                window,
                server.current_time,
                rng,
            )
            routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)

            # Occupy the drone
            server.agent_prop_set[dn].at_depot = False
            server.agent_prop_set[dn].current_package = pkg
            server.agent_task_allocation[dn] = (pkg, float('inf'))

            if csv_logger:
                csv_logger.log("drone_assignment.csv", {
                    "trial": trial_id, "time": time_step,
                    "drone_id": dn, "pkg_id": pkg,
                    "agent_tw_earliest_time": server.agent_task_windows[(dn, pkg)][0],
                    "agent_tw_latest_time":   server.agent_task_windows[(dn, pkg)][1],
                    "agent_tw_avail":         server.agent_task_windows[(dn, pkg)][2],
                    "true_return_time": rt, "true_delivery_time": td,
                    "success_prob": final_assignment[dn][1],
                    "approx_travel_time": routing_sim.active_packages[pkg].approx_travel_times.get(server.agent_set[dn].depot_number),
                    "true_travel_time": rt - td
                })

