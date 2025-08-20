
# routing_baselines.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Set

# Allocation and simulator types
from solver.scoba_types import MODE, GenericAllocation as RoutingAllocation
from .routing_types import RoutingSimulator, EuclideanLatLongMetric, convert_to_vector
from .routing_simulator import sample_true_delivery_return_time, travel_time_mean_minutes
from domains.routing.mcts import RoutingMCTSMDP
import logging



@dataclass
class RoutingDroneState:
    drone_idx: int
    pkg_assignment: int  # -1 return, 0 depot, >0 en route
    current_time: int
    true_delivery_return: Tuple[float, float]


def expected_hungarian(server: RoutingAllocation, routing_sim: RoutingSimulator, rng: Any = None,
                       csv_logger=None, trial_id=None, time_step=None) -> None:
    """
    Assign available drones to packages via Hungarian on negative success probability.
    """
    available_drones = [dn for dn, dp in server.agent_prop_set.items() if dp.at_depot]
    available_pkgs = list(routing_sim.active_packages.keys())
    n_d = len(available_drones)
    n_p = len(available_pkgs)
    if n_d == 0 or n_p == 0:
        return

    cost = np.ones((n_d, n_p), dtype=float)
    logging.info(f"Distnace threshold: {routing_sim.distance_thresh}")
    for i, dn in enumerate(available_drones):
        depot_loc = server.agent_set[dn].depot_loc
        for j, pkg in enumerate(available_pkgs):
            pkg_loc = routing_sim.active_packages[pkg].delivery
            dist = EuclideanLatLongMetric().evaluate(
                convert_to_vector(depot_loc), convert_to_vector(pkg_loc)
            )
            if dist <= routing_sim.distance_thresh:
                tw = server.agent_task_windows[(dn, pkg)]
                travel_time = tw[2] - tw[1]
                mean = travel_time
                std = mean / routing_sim.tt_est_std_scale
                from scipy.stats import norm  # approximate Epanechnikov
                succ_prob = norm(loc=mean, scale=std).cdf(tw[1])
                cost[i, j] = -succ_prob * routing_sim.delivery_reward

    if cost.min() == 1.0:
        return

    row_ind, col_ind = linear_sum_assignment(cost)
    for i, pj in zip(row_ind, col_ind):
        if pj < n_p and cost[i, pj] < 1.0:
            dn = available_drones[i]
            pkg = available_pkgs[pj]
            server.agent_task_allocation[dn] = (pkg, float('inf'))
            td = sample_true_delivery_return_time(
                server.agent_task_windows[(dn, pkg)],
                server.current_time,
                routing_sim.tt_est_std_scale,
                rng,
            )
            routing_sim.true_delivery_return[(dn, pkg)] = td
            server.agent_prop_set[dn].at_depot = False
            server.agent_prop_set[dn].current_package = pkg
            routing_sim.busy_packages[pkg] = routing_sim.active_packages.pop(pkg)
            routing_sim.num_active_packages -= 1
            if csv_logger:
                drone = server.agent_set[dn]
                loc = drone.depot_loc  # or another location if available
                csv_logger.log(
                    trial=trial_id,
                    time=time_step,
                    drone_id=dn,
                    pkg_id=pkg,
                    at_depot=False,
                    lat=loc.lat,
                    lon=loc.lon,
                    reward=routing_sim.delivery_reward,
                    success_prob=-cost[i, pj] / routing_sim.delivery_reward,
                    travel_time=td[0]  # delivery time from sample
                )


def earliest_due_date(server: RoutingAllocation, routing_sim: RoutingSimulator, rng: Any = None,
                      csv_logger=None, trial_id=None, time_step=None, comms_dict=None) -> None:
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
    for depot_number, depot_drones in available_drones.items():

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
            server.agent_prop_set[drone_id].interaction_events.sort(key=lambda ev: ev.timestamps[MODE.FINISH])
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

                    server.agent_task_allocation[drone_id] = (pkg_nm, float('inf'))  # inf for now for true delivery, can be updated later
                    # get the actual delivery time and return time
                    window = routing_sim.active_packages[pkg_nm].time_window
                    delivery_location = routing_sim.active_packages[pkg_nm].delivery
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
                    # break  # Only assign one package per drone
        




def get_current_routing_mcts_state(mdp: RoutingMCTSMDP, idx: int) -> RoutingDroneState:
    return RoutingDroneState(
        drone_idx=idx,
        pkg_assignment=mdp.drone_pkg_assignment[idx-1],
        current_time=int(mdp.server.current_time),
        true_delivery_return=mdp.true_delivery_return[idx-1],
    )
