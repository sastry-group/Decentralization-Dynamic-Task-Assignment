# routing_simulator.py
import math
import argparse
# from anyio import current_time

import numpy as np
from scipy.stats import uniform
from sklearn.neighbors import BallTree
from typing import Tuple
from typing import Any, List
import logging

from .routing_types import LatLonCoords, Package, CurrDroneSiteLocs, CityParams, parse_city_params
from .routing_types import convert_to_vector, EuclideanLatLongMetric
from solver.scoba_types import InteractionEvent, MODE

rng_int = np.random.default_rng(1345)

def get_travel_time_estimate(halton_nn_tree: BallTree,
                             loc1: LatLonCoords,
                             loc2: LatLonCoords,
                             estimate_matrix: np.ndarray,
                             time_scale: float, csv_logger=None) -> float:
    """
    Lookup nearest neighbor travel time estimate, fallback to Euclidean distance / speed.
    """
    v1 = convert_to_vector(loc1)
    v2 = convert_to_vector(loc2)
    # _, idx1 = halton_nn_tree.query([v1], k=1)
    # _, idx2 = halton_nn_tree.query([v2], k=1)
    # i1, i2 = idx1[0][0], idx2[0][0]
    # tt = estimate_matrix[i1, i2] / 60  #this is in seconds, convert to minutes
    dist_km = EuclideanLatLongMetric().evaluate(v1, v2)
    # Assume avg_speed = 0.00777 km/sec (i.e., 7.77 m/s)
    speed_km_per_min = 0.00777 * 60  # ~0.4662 km/min

    travel_time_min = math.ceil(dist_km / speed_km_per_min)
    # if tt == 0.0:
    #     dist = EuclideanLatLongMetric().evaluate(v1, v2)
    #     tt = dist / 0.00777
    # csv_logger.log("travel_time_estimates.csv", {
    #     "from_lat": loc1.lat,
    #     "from_lon": loc1.lon,
    #     "to_lat": loc2.lat,
    #     "to_lon": loc2.lon,
    #     "estimate": travel_time_min,
    # })
    # print(f"Travel time estimate from {loc1} to {loc2} output:  {tt / time_scale}")
    return travel_time_min


def generate_package_request(pkg_name, lat_dist: uniform,
                             lon_dist: uniform,
                             current_time: float,
                             tw_duration: float,
                             rng: np.random.Generator, 
                             depot_locs: List[LatLonCoords],
                             dist_thresh: float,
                             csv_logger=None) -> Package:
    """
    Create a random package delivery request.
    """

    lat = lat_dist.rvs(random_state=rng)
    lon = lon_dist.rvs(random_state=rng)
    delivery = LatLonCoords(lat=lat, lon=lon)
    start = current_time + rng_int.integers(tw_duration // 2, tw_duration + 1)
    length = rng_int.integers(tw_duration // 2, tw_duration + 1)
    window = (start, start + length)
    pkg = Package(delivery=delivery, time_window=window)
    return pkg
    # while True:
    #     lat = lat_dist.rvs(random_state=rng)
    #     lon = lon_dist.rvs(random_state=rng)
    #     delivery = LatLonCoords(lat=lat, lon=lon)

    #     for depot in depot_locs:
    #         dist = EuclideanLatLongMetric().evaluate(
    #             convert_to_vector(delivery),
    #             convert_to_vector(depot)
    #         )
    #         if dist <= (dist_thresh - 0.4):
    #             start = current_time + rng_int.integers(tw_duration // 2, tw_duration + 1)
    #             length = rng_int.integers(tw_duration // 2, tw_duration + 1)
    #             window = (start, start + length)
    #             return Package(delivery=delivery, time_window=window)







def epanechnikov(rng, mean, scale):
    # use acceptance–rejection to sample Epanechnikov(-1,1) then scale+shift
    sqrt5 = 5 ** 0.5
    while True:
        u = rng.uniform(-sqrt5, sqrt5)
        if rng.uniform() <= 0.75 * (1 - (u / sqrt5) ** 2):  # normalize for correct shape
            return mean + u * scale


def sample_true_delivery_return_time(drone_pkg_window: Tuple[float,float,float],
                                     current_time: float,
                                     std_scale: float,
                                     rng: np.random.Generator) -> Tuple[float,float]:
    """
    Sample true delivery and return times with uncertainty.
    """
    # print(f"Sampling true delivery and return time for window {drone_pkg_window} at current time {current_time}")
    travel_time = drone_pkg_window[2] - drone_pkg_window[1]
    # print(f"Travel time estimate: {travel_time} minutes")
    mean = travel_time 
    scale = travel_time / std_scale 
    # delta = rng.normal(loc=mean, scale=scale)
    # td = max(current_time + delta, drone_pkg_window[0])
    # rt = math.ceil(td) + rng.normal(loc=mean, scale=scale)
    ep = epanechnikov(rng, mean, scale)  # sample from Epanechnikov distribution
    td = math.ceil(current_time + ep)

    # print(f"current {current_time}, drone window start {drone_pkg_window[0]}, travel mean {mean}, epanechnikov output {ep} with scale {scale}, td {td}")
    td = max(td, drone_pkg_window[0]) # ensuring the delivery time is not before the start of the time window
    rt = td + math.ceil(ep)


    return td, rt   


def setup_routing_sim(params_fn: str,
                      halton_nn_tree: BallTree,
                      estimate_matrix: np.ndarray,
                      num_init_requests: int = 5,
                      new_request_prob: float = 0.75,
                      delivery_reward: float = 1000.0,
                      time_window_duration: float = 30.0,
                      in_transit_packages: dict[str, Package] = None,
                      rng: np.random.Generator = None,
                      depot_locs: List[LatLonCoords] = None,
                      csv_logger=None) -> 'RoutingSimulator':
    """
    Initialize a routing simulator environment with random initial packages.
    """
    if rng is None:
        rng = np.random.default_rng()
    city = parse_city_params(params_fn)
    lat_dist = uniform(loc=city.lat_start, scale=city.lat_end - city.lat_start)
    lon_dist = uniform(loc=city.lon_start, scale=city.lon_end - city.lon_start)

    active_packages = {}
    for n in range(1, num_init_requests+1):
        name = f"pkg{n}"
        # improve this to get te eactual distance_threshh
        pkg = generate_package_request(name, lat_dist, lon_dist, 0.0, time_window_duration, rng, depot_locs=depot_locs, dist_thresh=5 ,csv_logger=csv_logger)
        active_packages[name] = pkg


    from .routing_types import RoutingSimulator
    return RoutingSimulator(
        current_time=0.0,
        city_params=city,
        new_request_prob=new_request_prob,
        time_window_duration=time_window_duration,
        halton_nn_tree=halton_nn_tree,
        estimate_matrix=estimate_matrix,
        delivery_reward=delivery_reward,
        active_packages=active_packages,
        num_total_packages=num_init_requests,
        num_active_packages=num_init_requests,
        in_transit_packages=in_transit_packages,
        depot_locs=depot_locs if depot_locs else {},
    )


def update_routing_sim(sim, server, rng: np.random.Generator = None, csv_logger=None) -> None:
    """
    Advance the routing simulator by one timestep.
    """
    logging.debug(f"[Time Step] Advancing to t={sim.current_time}")
    if rng is None:
        rng = np.random.default_rng()
    old_time = sim.current_time
    
    sim.current_time += 1
    server.current_time += 1

    drone_locs = []
    site_locs = []

    to_remove_alloc = set()
    to_remove_busy = set()
    to_remove_done = set()

    # print(f"Current time: {sim.current_time}, Active packages: {sim.num_active_packages}, Total packages: {sim.num_total_packages}")
    for dn, (pkg, _) in list(server.agent_task_allocation.items()):
        delivery, ret = sim.true_delivery_return[(dn, pkg)]
        logging.info(f"Drone {dn} has package {pkg} with true delivery {delivery} and return {ret}, the window is {server.agent_task_windows[(dn, pkg)]}")

        # triggering only if deliveries are happening in this time step
        # For pickup, assign package to drone and mark package inactive
        if old_time < delivery <= sim.current_time:
            # print("Checking, sim time", sim.current_time, "delivery time", delivery)
            dp = server.agent_prop_set[dn]
            dp.current_package = ""
            dp.at_depot = False  
            # print("sim", sim.busy_packages[pkg])
            if delivery <= sim.busy_packages[pkg].time_window[1]:
                on_time = True
                sim.delivered_packages += 1
                drone_locs.append((sim.busy_packages[pkg].delivery, 'green'))
                logging.info(f"[Delivery] t={sim.current_time} | Drone {dn} delivered pkg {pkg} on time")
            else:
                sim.late_packages += 1
                on_time = False
                drone_locs.append((sim.busy_packages[pkg].delivery, 'red'))
                logging.info(f"[Delivery] t={sim.current_time} | Drone {dn} delivered pkg {pkg} late")
            to_remove_busy.add(pkg)
            
            if csv_logger:
                # pkg_window = server.agent_task_windows[(dn, pkg)]
                csv_logger.log("final_deliveries.csv", {
                    "trial": getattr(sim, "trial_id", None),
                    "timestep": sim.current_time,
                    "drone_id": dn,
                    "pkg_id": pkg,
                    "actual_delivery_time": delivery,
                    "deadline": sim.busy_packages[pkg].time_window[1],
                    "on_time": on_time,
                    # "window_start": pkg_window[0],
                    # "window_success": pkg_window[2],
                    # "depot_lat": server.agent_set[dn].depot_loc.lat,
                    # "depot_lon": server.agent_set[dn].depot_loc.lon,
                    # "delivery_lat": sim.busy_packages[pkg].delivery.lat,
                    # "delivery_lon": sim.busy_packages[pkg].delivery.lon,
                })

            sim.done_packages[pkg] = Package(
                delivery=sim.busy_packages[pkg].delivery,
                time_window=sim.busy_packages[pkg].time_window)
            logging.debug(f"sim done packages{sim.done_packages[pkg]}")
            sim.sum_of_delivery_time += (delivery - server.agent_task_windows[(dn, pkg)][0])
            logging.debug(
                f"[Delivery] t={sim.current_time} | Drone {dn} delivered pkg {pkg} "
                f"at {delivery:.2f}, deadline={sim.busy_packages[pkg].time_window[1]:.2f}, "
                f"on_time={delivery <= sim.busy_packages[pkg].time_window[1]}"
            )
        # For dropoff, free up drone and setup deletion of (drone,package) keys
        elif old_time < ret <= sim.current_time:
            server.agent_prop_set[dn].at_depot = True
            to_remove_alloc.add((dn, pkg))
            drone_locs.append((server.agent_set[dn].depot_loc, 'blue'))
            to_remove_done.add(pkg)
            logging.debug(f"[Return] t={sim.current_time} | Drone {dn} returned after delivering {pkg} at ret={ret:.2f}")
        else:
            # if pkg in sim.busy_packages or pkg in sim.done_packages:
            if pkg not in sim.busy_packages and pkg not in sim.done_packages:
                logging.warning(f"[t={sim.current_time}] Package {pkg} not found in busy or done — skipping.")
                continue
            depot = server.agent_set[dn].depot_loc
            
            if sim.current_time < delivery:
                pp_loc = sim.busy_packages[pkg].delivery

                max_diff = max(delivery, ret - delivery)
                interp_factor = (delivery - sim.current_time) / max_diff
                new_lat = depot.lat + (1 - interp_factor) * (pp_loc.lat - depot.lat)
                new_lon = depot.lon + (1 - interp_factor) * (pp_loc.lon - depot.lon)
            else:
                if pkg not in sim.done_packages:
                    logging.warning(f"[t={sim.current_time}] Package {pkg} not found in done_packages — skipping.")
                    continue

                pp_loc = sim.done_packages[pkg].delivery

                max_diff = max(delivery, ret - delivery)
                interp_factor = (ret - sim.current_time) / max_diff
                new_lat = depot.lat + interp_factor * (pp_loc.lat - depot.lat)
                new_lon = depot.lon + interp_factor * (pp_loc.lon - depot.lon)
                # pp_loc = sim.done_packages[pkg].delivery
                # interp_factor = (ret - sim.current_time)/(ret - delivery)
                # new_lat = depot.lat + (interp_factor)*(pp_loc.lat - depot.lat)
                # new_lon = depot.lon + (interp_factor)*(pp_loc.lon - depot.lon)

            drone_locs.append((LatLonCoords(lat=new_lat, lon=new_lon), 'blue'))


    # Delete bookkeeping keys for dropped off package
    for dn, pkg in to_remove_alloc:
        del sim.true_delivery_return[(dn, pkg)]
        del server.agent_task_windows[(dn, pkg)]
        del server.agent_task_allocation[dn]
    for pkg in to_remove_busy:
        del sim.busy_packages[pkg]
    for pkg in to_remove_done:
        if pkg in sim.done_packages:
            del sim.done_packages[pkg]

    expired = [p for p, rp in sim.active_packages.items() if rp.time_window[1] <= sim.current_time]
    for p in expired:
        logging.debug(f"[Expired] t={sim.current_time} | Package {p} expired at {sim.active_packages[p].time_window[1]}")
        sim.late_packages += 1
        site_locs.append((sim.active_packages[p].delivery, 'red'))
        del sim.active_packages[p]
        sim.num_active_packages -= 1

    # if rng.random() <= sim.new_request_prob:
    #     name = f"pkg{sim.num_total_packages + 1}"
    #     pkg = generate_package_request(
    #         name,
    #         uniform(loc=sim.city_params.lat_start, scale=sim.city_params.lat_end - sim.city_params.lat_start),
    #         uniform(loc=sim.city_params.lon_start, scale=sim.city_params.lon_end - sim.city_params.lon_start),
    #         sim.current_time,
    #         sim.time_window_duration,
    #         rng,
    #         depot_locs=sim.depot_locs, 
    #         dist_thresh= sim.distance_thresh,
    #         csv_logger=csv_logger
    #     )
    #     # logging.info(f"New package request generated: {pkg}")
    #     sim.num_total_packages += 1
    #     sim.num_active_packages += 1
    #     sim.active_packages[name] = pkg

    for p, rp in sim.active_packages.items():
        site_locs.append((rp.delivery, 'grey'))
    for p, rp in sim.busy_packages.items():
        site_locs.append((rp.delivery, 'grey'))

    sim.curr_drone_site_locs = CurrDroneSiteLocs(drone_locs, site_locs)
    logging.debug(f"[Time Step] Updated drone locations: {len(drone_locs)} drones")



def update_time_windows(sim, server, csv_logger=None) -> None:
    """
    Recompute interaction events and time windows for all active packages.
    """
    assert sim.current_time == server.current_time
    # print(f"[Time Step] Updating time windows at t={sim.current_time}, server time={server.current_time}")
    for dn, dp in server.agent_prop_set.items():
        events = dp.interaction_events # initialize events
        depot = server.agent_set[dn].depot_loc # latitude and long. of depot
        for pkg, rp in sim.active_packages.items(): # rp delivery location and time window
            key = (dn, pkg)
            if key not in server.agent_task_windows: # server.agent_task_windows is a dict of (dn, pkg) -> (start, finish, return to depot time)
                # compute the anticipated return time
                return_time = get_travel_time_estimate(
                    sim.halton_nn_tree,
                    rp.delivery,
                    depot,
                    sim.estimate_matrix,
                    sim.time_scale,
                    csv_logger=csv_logger
                )

                # this one is from sim.active_packages
                start, finish = rp.time_window   # earliest time the drone is allowed to depart the depot,  latest time the drone is allowed to finish delivery
                # now i am copying this package service window for the drone and what would be the latest return to base time
                ts = (start, finish, math.ceil(finish) + return_time) 
                server.agent_task_windows[key] = ts
                
                # logging.info(f"Creating interaction event for drone {dn} and package {pkg} with timestamps {ts}")
                ie = InteractionEvent(
                    agent_name=dn,
                    task_name=pkg,
                    timestamps={
                        MODE.START:   start,
                        MODE.FINISH:  finish,
                        MODE.RETURN: math.ceil(finish) + return_time,
                    }
                )
                events.append(ie)
                csv_logger.log("interaction_events.csv", {
                    "timestep": sim.current_time,
                    "drone": dn,
                    "package": pkg,
                    "earliest_time_attempt": start,
                    "latest_time_attemp": finish,
                    "success_return": math.ceil(finish) + return_time,
                })

        # now sort by the return timestamp
        events.sort(key=lambda e: e.timestamps[MODE.RETURN])
        dp.interaction_events = events
        # logging.info(f"[t={sim.current_time}] Drone {dn} updated interaction events: {len(events)}")



def parse_routing_commandline() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routing simulation CLI")
    p.add_argument("--trials", "-n", type=int, default=100)
    p.add_argument("--timesteps", "-t", type=int, default=500)
    p.add_argument("--n_drones", "-d", type=int, required=True)
    p.add_argument("--n_depots", "-e", type=int, required=True)
    p.add_argument("--new_request_prob", "-p", type=float, required=True)
    p.add_argument("--time_window", "-w", type=float, required=True)
    p.add_argument("baseline", choices=["dispatch","hungarian","css"], help="Method")
    p.add_argument("out_file_name")
    return p.parse_args()
