import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import deque, Counter
import logging
import heapq
import random

from domains.routing.routing_scoba import delivery_success_prob_common, cdf_travel_time
from solver.scoba_types import InteractionEvent, MODE, GenericAllocation as RoutingAllocation
from domains.routing.routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
from domains.routing.routing_simulator import sample_true_delivery_return_time, travel_time_mean_minutes
from domains.graph_builder import build_comm_structure




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



def delivery_util(reward: float, ref_time: float, ie: InteractionEvent) -> float:

    
    # Compute a realistic utility as expected reward
    p_succ = delivery_success_prob(std_scale=2.0, 
                                   ref_time=ref_time,
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
    p_i = delivery_success_prob(
        std_scale=routing_sim.tt_est_std_scale,
        ref_time=routing_sim.current_time,
        ie=ie
    )
    profile = dict(assigned)
    profile[agent] = pkg
    competitors = [n for n, p in assigned.items() if p == pkg and n != agent]
    
    # estimate competitors' probabilities for same pkg using their interaction_events

    p_sum = p_i
    for n in competitors:
        # find that neighbor's ie for this pkg
        p_n = 0.0
        for ie_n in server.agent_prop_set[n].interaction_events:
            if ie_n.task_name == pkg:
                p_n = delivery_success_prob(
                    std_scale=routing_sim.tt_est_std_scale,
                    ref_time=routing_sim.current_time,
                    ie=ie_n
                )
                break
        p_sum += p_n

    # "expected share of winning" proxy
    win_share = (p_i / p_sum) if p_sum > 0 else 1.0
    u = delivery_reward * p_i * win_share
    if competitors:
        u -= lambda_conflict



    logging.info(f"The win share for {agent} on pkg {pkg} is {win_share}, u {u} with p_i={p_i} and p_sum={p_sum} with competitors {competitors}")
    return u, len(competitors)

    



def delivery_success_prob(std_scale: float, ref_time: float, ie: InteractionEvent) -> float:
    travel_time = ie.travel_time # some estimate  trvel time
    mean = travel_time
    scale = travel_time / std_scale
    x = ie.timestamps[MODE.FINISH] - ref_time  # this is counting what the current time step is 

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




def iterative_best_response(server: RoutingAllocation, routing_sim: RoutingSimulator, rng=None, csv_logger=None,
                            init_method = "greedy", trial_id=None, time_step=None, comms_dict=None, allow_overlap=False, 
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

    # # Helper: get visible drones for a given agent - REMOVE
    # def get_visible_drones_for_agent(agent_id: str) -> List[str]:
    #     dnum = server.agent_set[agent_id].depot_number
    #     return visible_drones_by_depot.get(dnum, [])

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
    

    # --- Initial assignment (per depot)
    assigned: Dict[str, Optional[str]] = {}
    best_ies: Dict[str, Tuple[str, float]] = {}
    final_assignment: Dict[str, Tuple[str, float, InteractionEvent]] = {}

    for depot_num in depot_order:
        drones = depots[depot_num]
        assigned_per_depot = []
        
        if init_method == "empty":
            # No pre-assignments; best-response will fill in during k-rounds
            for dn in drones:
                best_ies[dn] = (None, 0.0)
        elif init_method == "random":
            for dn in drones:

                events = interaction_events_by_drone.get(dn, [])   
            if len(events) == 0:
                best_ies[dn] = (None, 0.0)
                continue         
            if allow_overlap is False:
                available_events = [ie for ie in events if ie.task_name not in assigned_per_depot and ie.task_name not in routing_sim.busy_packages]
            else:
                available_events = [ie for ie in events if ie.task_name not in assigned_per_depot and ie.task_name not in previously_pkgs_visible_to_depot[depot_num]]

            # safeguard
            if len(available_events) == 0:
                best_ies[dn] = (None, 0.0)
                continue

            ie = rng.choice(available_events)
            assigned[dn] = ie.task_name
            assigned_per_depot.append(ie.task_name)
            best_ies[dn] = (ie.task_name, 0.0)
            final_assignment[dn] = (ie.task_name, 0.0, ie)

        elif init_method == "greedy":
            for dn in drones:
                best_ie = None
                best_util = float("-inf")
                events = interaction_events_by_drone.get(dn, [])
                for ie in events:
                    if allow_overlap is False:
                        if ie.task_name in assigned_per_depot or ie.task_name in routing_sim.busy_packages:
                            continue
                    else:
                        if ie.task_name in assigned_per_depot or ie.task_name in previously_pkgs_visible_to_depot[depot_num]:
                            continue
                    u = delivery_util(routing_sim.delivery_reward, routing_sim.current_time, ie)
                    logging.info(f"[IBR] init greedy depot={depot_num} drone={dn} candidate={ie.task_name} util={u:.3f}")
                    if u > best_util:
                        best_util = u
                        best_ie = ie
                if best_ie:
                    logging.info(f"[IBR] init greedy depot={depot_num} drone={dn} selected={best_ie.task_name} util={best_util:.3f}")
                    assigned[dn] = best_ie.task_name
                    assigned_per_depot.append(best_ie.task_name)
                    best_ies[dn] = (best_ie.task_name, best_util)
                    final_assignment[dn] = (best_ie.task_name, best_util, best_ie)
                else:
                    best_ies[dn] = (None, 0.0)
        logging.info(f"[IBR] initial assigned: { {dn: assigned.get(dn) for dn in sorted(assigned)} }")      
    # --- Iterative best response (GLOBAL), information-aware ---
    # Important: each drone "sees" only drones from depots in its comms neighborhood.
    k_rounds = 20
    # Ensure iteration over depots.values() follows depot_order and is deterministic
    depots = {d: sorted(depots[d]) for d in depot_order if d in depots}
    all_considered_drones = [dn for drones in depots.values() for dn in drones]

    rounds_completed = 1
    steps_total = 0
    changes_total = 0
    num_agent_updates = 0
    num_candidate_evals = 0
    sum_visible_deg = 0
    num_visible_samples = 0
    sum_competitors_counted = 0
    sum_eval_work = 0

    for r in range(1, k_rounds + 1):
        changes_this_round = 0
        committed = set() 
        # Only drones at depot can re-choose
        for drone in all_considered_drones:
            if not server.agent_prop_set[drone].at_depot:
                continue

            steps_total += 1
       

            # visible_drones = [other for other in get_visible_drones_for_agent(drone)]
            visible_drones = visible_by_drone.get(drone, [])
            # if not visible_drones:
            #     continue

            competitor_drones = [
                n for n in visible_drones
                if (n in committed) or (not server.agent_prop_set[n].at_depot)
            ]
            num_agent_updates += 1
            sum_visible_deg += len(competitor_drones)
            num_visible_samples += 1
            # assigned_visible = {n: p for n, p in assigned.items() if n in visible_drones and p is not None}
            # assigned_visible = {n: assigned[n] for n in visible_drones if assigned.get(n) is not None}
            assigned_visible = {n: assigned[n] for n in competitor_drones if assigned.get(n) is not None}
            visible_assigned_pkgs = set(assigned_visible.values())

            # include previously visible packages for this drone's depot
            dnum = server.agent_set[drone].depot_number  
            prev_blocked = previously_pkgs_visible_to_depot.get(dnum, set())       
            visible_assigned_pkgs |= previously_pkgs_visible_to_depot.get(dnum, set())
            if allow_overlap:
                blocked_pkgs = set(prev_blocked)         
            else:
                blocked_pkgs = set(assigned_visible.values()) | set(prev_blocked) | set(routing_sim.busy_packages)

            logging.info(
                f"[IBR] BR t={time_step} drone={drone} depot={dnum} "
                f"sees_drones={visible_drones} "
                f"assigned_visible={assigned_visible} "
                f"prev_visible_pkgs={sorted(previously_pkgs_visible_to_depot.get(dnum,set()))} "
                f"prev_blocked={sorted(prev_blocked)} "
                f"blocked_pkgs={sorted(blocked_pkgs)}"
                f"to check diff visible_assigned_pkgs={sorted(visible_assigned_pkgs)}"
            )

            
            best_ie = None
            idle_util = 0.0
            # best_util = float("-inf")
            best_util = idle_util

            for ie in interaction_events_by_drone.get(drone, []):
                pkg = ie.task_name



                # and then:
                if pkg in blocked_pkgs:
                    continue

                # logging.info(f"[DBG] eval drone={drone} pkg={pkg} others_on_pkg={[n for n,p in assigned_visible.items() if p==pkg]}")


                u, comp_ct = compute_utility(
                    ie=ie,
                    agent=drone,
                    assigned=assigned_visible,
                    server=server,
                    routing_sim=routing_sim,
                    mode="exact",
                    lambda_conflict=500,
                    delivery_reward=routing_sim.delivery_reward
                )
                logging.info(f"[IBR] BR choice drone={drone} candidate={ie.task_name} util={u:.3f}")
                num_candidate_evals += 1
                sum_competitors_counted += comp_ct
                sum_eval_work += 1 + comp_ct 
                # deterministic tie-break to keep runs reproducible
                # if (u > best_util) or (u == best_util and (best_ie is None or ie.task_name < best_ie.task_name)):

                #     best_util, best_ie = u, ie
                if u > best_util:
                    best_util, best_ie = u, ie
            
            if best_ie:
                logging.info(
                    f"[IBR] BR choice drone={drone} choose={best_ie.task_name} "
                    f"util={best_util:.3f} prev={assigned.get(drone)}"
                )
            else:
                logging.info(
                    f"[IBR] BR choice drone={drone} choose=None (no feasible candidate)"
                )

            # # Commit immediately if the best response changes the action
            # if best_ie and assigned.get(drone) != best_ie.task_name:
            #     assigned[drone] = best_ie.task_name
            #     final_assignment[drone] = (best_ie.task_name, best_util, best_ie)
            #     changes_this_round += 1
            #     changes_total += 1
            current_pkg = assigned.get(drone)

            new_pkg = best_ie.task_name if best_ie is not None else None

            if new_pkg != current_pkg:
                assigned[drone] = new_pkg
                
                if best_ie is not None:
                    final_assignment[drone] = (new_pkg, best_util, best_ie)
                changes_this_round += 1
                changes_total += 1
            committed.add(drone)

        if changes_this_round == 0:
            break  # early convergence after a FULL round with zero changes
        rounds_completed += 1

    

    csv_logger.log("computational_efficiency_metrics.csv", {
        "trial": trial_id,
        "time": time_step,
        "iterations": steps_total,
        "k_rounds": rounds_completed,
        "changes": changes_total,

        # NEW: complexity proxies
        "agent_updates": num_agent_updates,
        "candidate_evals": num_candidate_evals,
        "avg_visible_degree": (sum_visible_deg / max(1, num_visible_samples)),
        "sum_visible_degree": sum_visible_deg,
        "visible_samples": num_visible_samples,
        "sum_eval_work": sum_eval_work,
        "avg_eval_work": sum_eval_work / max(1, num_candidate_evals),
        "avg_competitors_per_eval": sum_competitors_counted / max(1, num_candidate_evals),
    })
    
    logging.info("[IBR] Final assignments are: " + str(assigned))

    for dn, pkg in assigned.items():
        
        if allow_overlap is False:
            if pkg in routing_sim.busy_packages:
                logging.warning(f"Drone {dn} lost allocation for {pkg} (already taken)")
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