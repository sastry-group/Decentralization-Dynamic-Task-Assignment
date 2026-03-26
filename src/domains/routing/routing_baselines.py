
# routing_baselines.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Set
from scipy.stats import norm  # approximate Epanechnikov

# Allocation and simulator types
from solver.scoba_types import MODE, GenericAllocation as RoutingAllocation
from .routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
# from .routing_simulator import sample_true_delivery_return_time, travel_time_mean_minutes
from .travel_model import  sample_true_delivery_return_time, delivery_success_prob

from domains.routing.mcts import RoutingMCTSMDP
import logging



@dataclass
class RoutingDroneState:
    drone_idx: int
    pkg_assignment: int  # -1 return, 0 depot, >0 en route
    current_time: int
    true_delivery_return: Tuple[float, float]


def expected_hungarian(server: RoutingAllocation, routing_sim: RoutingSimulator, rng: Any = None,
                       csv_logger=None, trial_id=None, time_step=None, comms_dict=None,
                       allow_overlap=False, depot_order="asc") -> None:
    """
    Per-comms-group Hungarian assignment on negative success probability.
    Mirrors IBR's comms/blocking setup; replaces global solve with one per depot group.
    """

    # ------------------------------------------------------------------ #
    # 1. Group drones by depot (same as IBR)                              #
    # ------------------------------------------------------------------ #
    depots: Dict[int, List[str]] = {}
    global_depos: Dict[int, List[str]] = {}
    for dn, dp in server.agent_prop_set.items():
        dnum = server.agent_set[dn].depot_number
        if dp.at_depot:
            depots.setdefault(dnum, []).append(dn)
        global_depos.setdefault(dnum, []).append(dn)

    if not any(depots.values()):
        logging.info(f"[Hungarian] t={time_step} no drones at depot; skipping.")
        return

    # ------------------------------------------------------------------ #
    # 2. Build comms neighborhoods (identical to IBR)                     #
    # ------------------------------------------------------------------ #
    all_depots_comm_net = set(depots.keys())
    global_depots_comm_net = set(global_depos.keys())

    if comms_dict is None:
        depot_neighbors = {d: set(all_depots_comm_net) for d in all_depots_comm_net}
        global_depots_neighbors = {d: set(global_depots_comm_net) for d in global_depots_comm_net}
    else:
        depot_neighbors = {d: {d} | set(comms_dict.get(d, [])) for d in all_depots_comm_net}
        global_depots_neighbors = {d: {d} | set(comms_dict.get(d, [])) for d in global_depots_comm_net}
        for d in comms_dict.keys():
            depot_neighbors.setdefault(d, {d} | set(comms_dict.get(d, [])))
            global_depots_neighbors.setdefault(d, {d} | set(comms_dict.get(d, [])))

    # ------------------------------------------------------------------ #
    # 3. Visible drones per depot (identical to IBR)                      #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 4. Previously-seen packages per depot (identical to IBR)            #
    # ------------------------------------------------------------------ #
    previously_pkgs_visible_to_depot: Dict[int, Set[str]] = {d: set() for d in global_depos}
    active_now = set(routing_sim.active_packages.keys())

    for depot_num in global_depos:
        visible_drones = set(global_visible_drones_by_depot.get(depot_num, []))
        prev: Set[str] = set()
        for pkg, claimants in getattr(routing_sim, "package_claims", {}).items():
            if pkg not in active_now:
                continue
            claimants = set(claimants) if not isinstance(claimants, set) else claimants
            if claimants & visible_drones:
                prev.add(pkg)
        for pkg, winner_dn in getattr(routing_sim, "package_winners", {}).items():
            if pkg in active_now and winner_dn in visible_drones:
                prev.add(pkg)
        previously_pkgs_visible_to_depot[depot_num] = prev

    # ------------------------------------------------------------------ #
    # 5. Depot ordering (identical to IBR)                                #
    # ------------------------------------------------------------------ #
    if depot_order == "asc":
        ordered_depots = sorted(depots.keys())
    elif depot_order == "desc":
        ordered_depots = sorted(depots.keys(), reverse=True)
    elif depot_order == "random":
        ordered_depots = rng.sample(list(depots.keys()), len(depots))
    else:
        ordered_depots = sorted(depots.keys())

    # tracks packages committed across depot solves (no-overlap mode)
    globally_assigned: Set[str] = set()

    # ------------------------------------------------------------------ #
    # 6. Per-depot Hungarian solve                                        #
    # ------------------------------------------------------------------ #
    for depot_num in ordered_depots:
        local_drones = depots[depot_num]           # drones actually at THIS depot
        depot_loc = server.agent_set[local_drones[0]].depot_loc
        prev_blocked = previously_pkgs_visible_to_depot.get(depot_num, set())

        # Packages in range of this depot
        in_range = [
            pkg for pkg, pp in routing_sim.active_packages.items()
            if EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
            ) <= routing_sim.distance_thresh
        ]

        # Available packages: exclude prev-blocked, busy, and already-claimed this round
        if allow_overlap:
            available_pkgs = [p for p in in_range if p not in prev_blocked and p not in globally_assigned]
        else:
            available_pkgs = [
                p for p in in_range
                if p not in prev_blocked
                and p not in routing_sim.busy_packages
                and p not in globally_assigned
            ]


        if not local_drones or not available_pkgs:
            logging.info(f"[Hungarian] depot {depot_num}: no drones or no packages, skipping.")
            continue

        logging.info(f"[Hungarian] depot {depot_num}: {len(local_drones)} drones, "
                     f"{len(in_range)} pkgs in range (pre-filter), "
                     f"{len(available_pkgs)} pkgs available (post-filter), "
                     f"prev_blocked={len(prev_blocked)}, "
                     f"globally_assigned={len(globally_assigned)}, "
                     f"busy={len(routing_sim.busy_packages)}")
        for dn in local_drones:
            logging.info(f"  drone={dn} depot_loc={server.agent_set[dn].depot_loc}")
        for pkg in available_pkgs[:5]:  # first 5 to avoid spam
            pp = routing_sim.active_packages[pkg]
            logging.info(f"  pkg={pkg} delivery_loc={pp.delivery}")


        n_d = len(local_drones)
        n_p = len(available_pkgs)

        # Build cost matrix: rows=drones, cols=packages
        # Default 1.0 = infeasible (no interaction event / out of range)
        cost = np.ones((n_d, n_p), dtype=float)

        for i, dn in enumerate(local_drones):
            for j, pkg in enumerate(available_pkgs):
                ie = next(
                    (e for e in server.agent_prop_set[dn].interaction_events
                     if e.task_name == pkg),
                    None
                )
                if ie is None:
                    continue

                # Use delivery_success_prob (matches Julia exactly):
                # mu  = timestamps[SUCCESS] - timestamps[FINISH]  (avail - latest)
                # cdf evaluated at timestamps[FINISH]             (latest time)
                succ_prob = delivery_success_prob(
                    ref_time=routing_sim.current_time,
                    ie=ie,
                    std_scale=routing_sim.tt_est_std_scale,
                )
                cost[i, j] = -succ_prob * routing_sim.delivery_reward

        if cost.min() >= 1.0:
            logging.info(f"[Hungarian] depot {depot_num}: all costs infeasible, skipping.")
            continue

        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1.0:
                globally_assigned.add(available_pkgs[j]) 

        logging.info(f"[Hungarian] depot {depot_num} assignments: "
                     f"{ {local_drones[i]: available_pkgs[j] for i, j in zip(row_ind, col_ind) if cost[i,j] < 1.0} }")

        # ------------------------------------------------------------------ #
        # 7. Finalize assignments  — mirrors IBR's finalization exactly       #
        # ------------------------------------------------------------------ #
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= 1.0:          # infeasible cell, skip
                continue

            dn  = local_drones[i]
            pkg = available_pkgs[j]
            succ_prob = -cost[i, j] / routing_sim.delivery_reward

            window            = routing_sim.active_packages[pkg].time_window
            depot_loc         = server.agent_set[dn].depot_loc
            delivery_location = routing_sim.active_packages[pkg].delivery

            td, rt = sample_true_delivery_return_time(
                depot_loc,
                delivery_location,
                window,
                server.current_time,
                rng,
            )

            if not allow_overlap:
                # ---- no-overlap branch (same as IBR allow_overlap=False) ----
                if pkg in routing_sim.busy_packages:
                    logging.warning(f"[Hungarian] Drone {dn} lost allocation for {pkg} (already taken)")
                    continue

                globally_assigned.add(pkg)
                server.agent_task_allocation[dn]        = (pkg, float('inf'))
                routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
                server.agent_prop_set[dn].at_depot       = False
                server.agent_prop_set[dn].current_package = pkg
                routing_sim.busy_packages[pkg] = routing_sim.active_packages.pop(pkg)
                routing_sim.num_active_packages -= 1

                if csv_logger:
                    csv_logger.log("drone_assignment.csv", {
                        "trial":                  trial_id,
                        "time":                   time_step,
                        "drone_id":               dn,
                        "pkg_id":                 pkg,
                        "agent_tw_earliest_time": server.agent_task_windows[(dn, pkg)][0],
                        "agent_tw_latest_time":   server.agent_task_windows[(dn, pkg)][1],
                        "agent_tw_avail":         server.agent_task_windows[(dn, pkg)][2],
                        "true_return_time":       rt,
                        "true_delivery_time":     td,
                        "success_prob":           succ_prob,
                        "approx_travel_time":     routing_sim.busy_packages[pkg].approx_travel_times.get(depot_num),
                        "true_travel_time":       rt - td,
                    })

            else:
                # ---- overlap branch (same as IBR allow_overlap=True) ----
                if pkg is None:
                    continue

                routing_sim.package_claims.setdefault(pkg, set()).add(dn)
                routing_sim.package_registry[pkg]["claimed_by"].append(dn)
                routing_sim.package_registry[pkg]["time_assigned"].append(time_step)
                routing_sim.busy_packages[pkg] = routing_sim.active_packages.get(pkg)

                routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
                server.agent_prop_set[dn].at_depot            = False
                server.agent_prop_set[dn].current_package     = pkg
                server.agent_task_allocation[dn]              = (pkg, float('inf'))

                if csv_logger:
                    csv_logger.log("drone_assignment.csv", {
                        "trial":                  trial_id,
                        "time":                   time_step,
                        "drone_id":               dn,
                        "pkg_id":                 pkg,
                        "agent_tw_earliest_time": server.agent_task_windows[(dn, pkg)][0],
                        "agent_tw_latest_time":   server.agent_task_windows[(dn, pkg)][1],
                        "agent_tw_avail":         server.agent_task_windows[(dn, pkg)][2],
                        "true_return_time":       rt,
                        "true_delivery_time":     td,
                        "success_prob":           succ_prob,
                        "approx_travel_time":     routing_sim.active_packages[pkg].approx_travel_times.get(depot_num),
                        "true_travel_time":       rt - td,
                    })



def earliest_due_date(server: RoutingAllocation, routing_sim: RoutingSimulator, rng: Any = None,
                      csv_logger=None, trial_id=None, time_step=None, comms_dict=None, allow_overlap=False, depot_order="asc") -> None:
    """
    Assign drones to earliest due packages within range.
    """
    # logging.info("Using EDD baseline for routing allocation.")

    available_drones: Dict[int, List[str]] = {}

    
    # group drones by depot
    for drone_nm, dp in server.agent_prop_set.items():
        drone = server.agent_set[drone_nm]
        if dp.at_depot is True:
            if drone.depot_number not in available_drones:
                available_drones[drone.depot_number] = [drone_nm]
            else:
                available_drones[drone.depot_number].append(drone_nm)

    all_assigned_pkgs: Set[str] = set()
    # logging.info(f"Drones at depot information: {depots}")


    # Finding nearby packages
    if depot_order == "random":
        depot_numbers = list(available_drones.keys())
        rng.shuffle(depot_numbers)
        order = ((d, available_drones[d]) for d in depot_numbers)

    elif depot_order == "desc":
        depot_numbers = sorted(available_drones.keys(), reverse=True)
        order = ((d, available_drones[d]) for d in depot_numbers)

    else:
        order= available_drones.items()

    for depot_number, depot_drones in order:
    # for depot_number, depot_drones in available_drones.items():

        # all the drones in this depot share the same depot location
        depot_loc = server.agent_set[depot_drones[0]].depot_loc

        pkgs_in_range: Set[str] = set()
        for pkg_nm, pp in routing_sim.active_packages.items():
            dist = EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
            )
            if dist <= routing_sim.distance_thresh:            
                pkgs_in_range.add(pkg_nm)

        # assigning drones to packages
        for drone_id in depot_drones:
            ie_idx = None
            #filtering interaction events based on in_range
            server.agent_prop_set[drone_id].interaction_events.sort(key=lambda ev: ev.timestamps[MODE.SUCCESS])
            for idx, ie in enumerate(server.agent_prop_set[drone_id].interaction_events):
                if (
                    ie.task_name not in all_assigned_pkgs
                    and ie.task_name in pkgs_in_range
                    and ie.task_name not in routing_sim.busy_packages
                ):
                    ie_idx = idx
                    break

            if ie_idx is not None:
                pkg_nm = server.agent_prop_set[drone_id].interaction_events[ie_idx].task_name
                # Assign the first valid package
                if pkg_nm not in routing_sim.busy_packages:
                    all_assigned_pkgs.add(pkg_nm)

                    # server.agent_task_allocation[drone_id] = (pkg_nm, float('inf'))  # inf for now for true delivery, can be updated later
                    # get the actual delivery time and return time
                    window = routing_sim.active_packages[pkg_nm].time_window
                    delivery_location = routing_sim.active_packages[pkg_nm].delivery


                    server.agent_prop_set[drone_id].current_package = pkg_nm
                    server.agent_task_allocation[drone_id] = (pkg_nm, float("inf"))
                    if allow_overlap:
                        routing_sim.package_claims.setdefault(pkg_nm, set()).add(drone_id)
                        routing_sim.package_registry[pkg_nm]["claimed_by"].append(drone_id)
                        routing_sim.package_registry[pkg_nm]["time_assigned"].append(time_step)
                        routing_sim.package_registry[pkg_nm]["winner"] = drone_id
                        routing_sim.package_winners[pkg_nm] = drone_id

                    td, rt = sample_true_delivery_return_time(
                        depot_loc,
                        delivery_location,
                        window,
                        server.current_time,
                        rng,
                    )
                    # Update sim state
                    routing_sim.true_delivery_return[(drone_id, pkg_nm)] = (td, rt)
                    server.agent_prop_set[drone_id].at_depot = False
                    # routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages.pop(pkg_nm)
                    # routing_sim.num_active_packages -= 1


                    if allow_overlap:
                        routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages[pkg_nm]
                    else:
                        routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages.pop(pkg_nm)
                        routing_sim.num_active_packages -= 1
                
                
                    if csv_logger:
                        csv_logger.log("drone_assignment.csv", {
                            "trial": trial_id,
                            "time": time_step,
                            "drone_id": drone_id,
                            "depot_number": depot_number,
                            "pkg_id": pkg_nm,
                            "pkg_earliest_time": server.agent_task_windows[(drone_id, pkg_nm)][0],
                            "pkg_latest_time": server.agent_task_windows[(drone_id, pkg_nm)][1],
                            "true_return_time": rt,
                            "true_delivery_time": td,
                            "approx_travel_time": routing_sim.busy_packages[pkg_nm].approx_travel_times.get(depot_number),
                            "true_travel_time": rt-td
                        })

                    logging.info(f"[Depot {depot_number}] Drone {drone_id} assigned package {pkg_nm}, package delivery time {td}, return time {rt}, package window (start, end, nominal) {server.agent_task_windows[(drone_id, pkg_nm)]}")

        




def get_current_routing_mcts_state(mdp: RoutingMCTSMDP, idx: int) -> RoutingDroneState:
    return RoutingDroneState(
        drone_idx=idx,
        pkg_assignment=mdp.drone_pkg_assignment[idx-1],
        current_time=int(mdp.server.current_time),
        true_delivery_return=mdp.true_delivery_return[idx-1],
    )
