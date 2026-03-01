# routing_simulator.py
import math
import argparse
# from anyio import current_time

import numpy as np
from scipy.stats import uniform
from sklearn.neighbors import BallTree

from typing import List, Dict
import logging

from .travel_model import delivery_success_prob_common, sample_true_delivery_return_time, travel_time_mean_minutes
from .routing_types import LatLonCoords, Package, CurrDroneSiteLocs, CityParams, parse_city_params
from .routing_types import convert_to_vector, EuclideanLatLongMetric
from solver.scoba_types import InteractionEvent, MODE

# TRAVEL = dict(
#     avg_speed_km_per_min = 0.00777 * 60 / 1.2, # ~0.4662 km/min , just a scale factor to icnrease travel times
#     cv = 0.33,           # stdev = cv * mean   (tune 0.2–0.4 to taste)
#     dist = "epanechnikov"  # "epanechnikov" or "normal"
# )

# def travel_time_mean_minutes(loc1: LatLonCoords, loc2: LatLonCoords) -> float:
#     v1, v2 = convert_to_vector(loc1), convert_to_vector(loc2)
#     dist_km = EuclideanLatLongMetric().evaluate(v1, v2)
#     mu = dist_km / TRAVEL["avg_speed_km_per_min"]
#     return max(math.ceil(mu), 3)  # keep your floor of 3 min



def generate_package_request(pkg_name, lat_dist: uniform, lon_dist: uniform,
                             current_time: float, tw_duration: float, 
                             rng: np.random.Generator,
                             depots: dict[str,List[LatLonCoords]],
                             dist_thresh: float,
                             csv_logger=None) -> Package:
    lat = lat_dist.rvs(random_state=rng)
    lon = lon_dist.rvs(random_state=rng)
    delivery = LatLonCoords(lat=lat, lon=lon)

    approx_travel_times = {}
    distance_dict = {}
    for depot_idx, depot in depots.items():
        mu_tt = travel_time_mean_minutes(depot.location, delivery)  # unified deterministic mean
        approx_travel_times[depot_idx] = mu_tt
        dist_km = EuclideanLatLongMetric().evaluate(depot.location, delivery)
        distance_dict[depot_idx] =  dist_km

    # Pick window length relative to nearest depot mean
    mu_nearest = min(approx_travel_times.values())
    k_low, k_high = 0.8, 1.4
    duration = round(rng.uniform(k_low * mu_nearest, k_high * mu_nearest))
    start = round(current_time + rng.uniform(tw_duration // 2, tw_duration))
    # duration = max(start, approx_travel_times[min(approx_travel_times, key=approx_travel_times.get)] + 2) # safeguard
    window = (start, start + duration)

    pkg = Package(
        name=pkg_name,
        delivery=delivery,
        time_window=window,
        approx_travel_times=approx_travel_times,
        distance_to_depots=distance_dict
    )

    if csv_logger:
        csv_logger.log("pkg_gen_info.csv",{
            "pkg_id": pkg_name,
            "lat": lat, "lon": lon,
            "start": window[0], "end": window[1],
            "duration": duration,
            **{f"mu_depot_{i}": mu for i, mu in approx_travel_times.items()},
            **{f"dist_depot_{i}": dist for i, dist in distance_dict.items()}
        })

    return pkg



# def sample_true_travel_time(mu: float, rng: np.random.Generator) -> float:
#     sigma = max(TRAVEL["cv"] * mu, 1e-6)
#     if TRAVEL["dist"] == "epanechnikov":
#         # u ~ Epanechnikov with Var(u)=1 using the sqrt(5) trick; std = sigma
#         sqrt5 = 5 ** 0.5
#         while True:
#             u = rng.uniform(-sqrt5, sqrt5)
#             if rng.uniform() <= 0.75 * (1 - (u / sqrt5) ** 2):
#                 return max(1.0, round(mu + u * sigma))
#     elif TRAVEL["dist"] == "normal":
#         return max(1.0, round(rng.normal(mu, sigma)))
#     else:
#         raise ValueError("Unknown TRAVEL['dist']")




# def sample_true_delivery_return_time(
#     depot_loc: LatLonCoords,
#     delivery_loc: LatLonCoords,
#     window: Tuple[float, float],
#     current_time: float,
#     rng: np.random.Generator
# ) -> Tuple[float, float]:
#     """
#     Sample true delivery time (arrive at customer) and true return time (back to depot),
#     using the same uncertainty model as everywhere else.
#     """
#     mu_out = travel_time_mean_minutes(depot_loc, delivery_loc)
#     mu_back = travel_time_mean_minutes(delivery_loc, depot_loc)

#     # sample both legs with the same distribution family & CV
#     tt_out = sample_true_travel_time(mu_out, rng)
#     tt_back = sample_true_travel_time(mu_back, rng)

#     # depart immediately; arrive at
#     td = round(current_time + tt_out)
#     # respect time window start: wait if early
#     td = max(td, window[0])

#     # return after (waiting does not reduce flight time)
#     rt = round(td + tt_back)

#     return td, rt



def setup_routing_sim(server, params_fn: str,
                      halton_nn_tree: BallTree,
                    
                      estimate_matrix: np.ndarray,
                      num_init_requests: int = 5,
                      new_request_prob: float = 0.75,
                      delivery_reward: float = 1000.0,
                      time_window_duration: float = 30.0,
                      in_transit_packages: dict[str, Package] = None,
                      rng: np.random.Generator = None,
                      depots: Dict[str, List[LatLonCoords]] = None,
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
    package_registry = {}
    for n in range(1, num_init_requests+1):
        name = f"pkg{n}"
        # improve this to get te eactual distance_threshh
        pkg = generate_package_request(name, lat_dist, lon_dist, 0.0, 
                                       time_window_duration, rng, 
                                       depots=depots, dist_thresh=5 ,
                                       csv_logger=csv_logger)
        active_packages[name] = pkg
        package_registry[name] = pkg
        package_registry[name] = {
            "obj": pkg,             
            "claimed_by": [],   
            "winner": None,         
            "time_assigned": [],
        }


    from .routing_types import RoutingSimulator
    sim = RoutingSimulator(
        current_time=0.0,
        city_params=city,
        new_request_prob=new_request_prob,
        time_window_duration=time_window_duration,
        halton_nn_tree=halton_nn_tree,
        estimate_matrix=estimate_matrix,
        delivery_reward=delivery_reward,
        active_packages=active_packages,
        package_registry=package_registry,
        num_total_packages=num_init_requests,
        num_active_packages=num_init_requests,
        in_transit_packages=in_transit_packages,
        depots=depots if depots else {},
    )
    for drone_nm, props in server.agent_prop_set.items():
        props.available_at = 0.0
        props.at_depot = True
        props.current_package = ""

    return sim


def update_routing_sim(trial, sim, server, rng: np.random.Generator = None, csv_logger=None,
                       allow_overlap=False) -> None:
    """
    Advance the routing simulator by one timestep.
    """
    logging.debug(f"[Time Step] Advancing to t={sim.current_time}")
    if rng is None:
        rng = np.random.default_rng()

    # First update the simulator and server time
    old_time = sim.current_time
    sim.current_time += 1
    server.current_time += 1

    curr_drone_locs_cols = []  # List[Tuple[LatLonCoords, str]]
    curr_sites_locs_cols = []  # List[Tuple[LatLonCoords, str]]

    # 2) Simulate any true pickups and dropoffs
    keys_to_del = set()            # Set[Tuple[str, str]]
    busy_packages_to_del = set()   # Set[str]
    done_packages_to_del = set()   # Set[str]

    # ---------- resolve package races at first moment of delivery ----------
    # For each contested package, find claimants with delivery in (old_time, sim.current_time]
    # If any, pick the earliest arrival; if tie, random winner among the tie.
    if allow_overlap:
        for pkg, claimants in list(sim.package_claims.items()):
            if pkg in sim.package_winners:
                continue  # already resolved earlier

            arrivals = []
            for dn in list(claimants):
                if (dn, pkg) not in sim.true_delivery_return:
                    continue
                td, rt = sim.true_delivery_return[(dn, pkg)]
                if (td > old_time) and (td <= sim.current_time):
                    arrivals.append((dn, td))

            if not arrivals:
                continue

            # earliest arrival time
            min_td = min(td for _, td in arrivals)
            earliest = [dn for dn, td in arrivals if td == min_td]
            if len(earliest) == 1:
                winner = earliest[0]
            else:
                winner = rng.choice(earliest)  # tie-break

            # print(f"Adding winner {winner} for package {pkg} at time {sim.current_time}")
            sim.package_winners[pkg] = str(winner)

            # # Deliver no

        # ---------- END NEW race resolution ----------
        # 2) Simulate any drone state transitions (either delivery interpolation or return)
        for drone_nm, v in list(server.agent_task_allocation.items()):
            package_nm, _ = v
            flag = "None"
            if (drone_nm, package_nm) not in sim.true_delivery_return:
                continue

            drone_props = server.agent_prop_set[drone_nm]
            depot_loc = server.agent_set[drone_nm].depot_loc
            delivery, return_time = sim.true_delivery_return[(drone_nm, package_nm)]
            pkg_obj = sim.package_registry[package_nm].get('obj')
            deadline = pkg_obj.time_window[1]
            winner = sim.package_winners.get(package_nm)
            
            if (delivery > old_time) and (delivery <= sim.current_time):

                if winner == drone_nm:
                    server.agent_prop_set[drone_nm].current_package = ""
                    server.agent_prop_set[drone_nm].at_depot = False  # For good measure
                    # Case A: drone reaches its own delivery instant this tick
                    if delivery <= deadline:
                        sim.delivered_packages += 1
                        sim.package_registry[package_nm]["winner"] = drone_nm
                        flag = "True"
                        curr_drone_locs_cols.append((pkg_obj.delivery, "green"))
                        logging.info(f"[WIN] {winner} delivered {package_nm} on-time at {delivery}")
                        # Visual at package location handled above; nothing else here.
                            
                    else:
                        # Late delivery: mark as late, but still heads back and stays busy till return_time
                        sim.late_packages += 1
                        # print(f"Package {package_nm} late by drone {drone_nm} at timestep {sim.current_time}")
                        drone_props.current_package = ""
                        drone_props.at_depot = False
                        flag = "False"
                        curr_drone_locs_cols.append((pkg_obj.delivery, "red"))
                        logging.info(f"{drone_nm} reached {package_nm} LATE at {delivery} (deadline {deadline})")

                    sim.active_packages.pop(package_nm, None)
                    if csv_logger:
                        csv_logger.log("final_deliveries.csv", {
                            "trial": trial,
                            "timestep": sim.current_time,
                            "drone_id": winner,
                            "pkg_id": package_nm,
                            "actual_delivery_time": delivery,
                            "deadline": deadline,
                            "on_time": flag,
                        })
                    
                    pkg_ref = sim.busy_packages.get(package_nm) or sim.package_registry[package_nm].get("obj")
                    if pkg_ref is not None:
                        sim.done_packages[package_nm] = pkg_ref
                    sim.package_claims.pop(package_nm, None)
                    busy_packages_to_del.add(package_nm)
                    logging.info(f"[CLEANUP] delivered pkg={package_nm} removed claims; claims_now={package_nm in sim.package_claims}")

                    
                    # sim.done_packages[package_nm] = sim.busy_packages[package_nm]

                    # Increment loss with difference from start of window
                    sim.sum_of_delivery_time += delivery - server.agent_task_windows[(drone_nm, package_nm)][0]

                    logging.debug(
                        f"[Delivery] t={sim.current_time} | Drone {drone_nm} delivered pkg {package_nm} "
                        f"at {delivery:.2f}, deadline={sim.busy_packages[package_nm].time_window[1]:.2f}, "
                        f"on_time={delivery <= sim.busy_packages[package_nm].time_window[1]}"
                    )                    


                else:
                    # Loser: arrived but found it gone; still heads back and stays busy till return_time
                    logging.info(f"[LOSE] {drone_nm} reached {package_nm} at {delivery} but it was already taken by {winner}")
                    # drone_props.current_package = "" #because it lost the package
                    # Optional: mark a different color at site to visualize "arrived too late"
                    curr_drone_locs_cols.append((pkg_obj.delivery, "orange"))
        

            # Case B: drone completes return this tick
            elif (return_time > old_time) and (return_time <= sim.current_time):
                drone_props.at_depot = True
                drone_props.available_at = return_time
                drone_props.current_package = ""
                keys_to_del.add((drone_nm, package_nm))
                done_packages_to_del.add(package_nm)
                curr_drone_locs_cols.append((depot_loc, "blue"))
                logging.info(f"{drone_nm} back at depot (after attempting {package_nm})")

            else:
                # Interpolate position for plotting (depot -> site or site -> depot)
                # Choose source of package location from done or registry/active
                if package_nm not in sim.done_packages:
                    continue
                depot_loc = server.agent_set[drone_nm].depot_loc

                if sim.current_time < delivery:
                    # outbound leg: depot -> site
                    pp_loc =pkg_obj.delivery
                    denom = max(delivery - old_time, 1)  # guard
                    # use proportional interpolation using how far we are from delivery
                    maxdiff = max(delivery, (return_time - delivery))
                    interp_factor = (delivery - sim.current_time) / maxdiff if maxdiff != 0 else 0.0
                    new_lat = depot_loc.lat + (1.0 - interp_factor) * (pp_loc.lat - depot_loc.lat)
                    new_lon = depot_loc.lon + (1.0 - interp_factor) * (pp_loc.lon - depot_loc.lon)
                else:
                    # inbound leg: site -> depot
                    pp_loc = sim.done_packages.get(package_nm).delivery
                    denom = (return_time - delivery)
                    interp_factor = (return_time - sim.current_time) / denom if denom != 0 else 0.0
                    new_lat = depot_loc.lat + (interp_factor) * (pp_loc.lat - depot_loc.lat)
                    new_lon = depot_loc.lon + (interp_factor) * (pp_loc.lon - depot_loc.lon)

                curr_drone_locs_cols.append((type(depot_loc)(lat=new_lat, lon=new_lon), "blue"))

        # Clean up per-drone allocation and timing after return
        for k in keys_to_del:
            sim.true_delivery_return.pop(k, None)
            server.agent_task_windows.pop(k, None)
            server.agent_task_allocation.pop(k[0], None)

        # Once a package is resolved, clear its claims
        # for pkg, winner in list(sim.package_winners.items()):
        #     # remove claims of drones that have returned; keep others until they return (harmless)
        #     sim.package_claims.pop(pkg, None)

        for r in busy_packages_to_del:
            sim.busy_packages.pop(r, None)

        for r in done_packages_to_del:
            sim.done_packages.pop(r, None)

        # 3) Handle active packages that expired without any winner
        packages_to_del = set()
        for package_nm, rp in list(sim.active_packages.items()):
            if rp.time_window[1] < sim.current_time:
                # No one delivered in time
                if package_nm not in sim.package_claims:
                    logging.info(f"{package_nm} expired without successful delivery")
                    sim.late_packages += 1
                    # print(f"Package {package_nm} expired without delivery at timestep {sim.current_time}")
                    packages_to_del.add(package_nm)
                    curr_sites_locs_cols.append((rp.delivery, "red"))


        for r in packages_to_del:
            sim.active_packages.pop(r, None)
            sim.num_active_packages -= 1

        # 5) Grey markers for sites:
        # active packages + (optional) packages with outstanding claims
        for pkg_nm, pp in sim.active_packages.items():
            curr_sites_locs_cols.append((pp.delivery, "grey"))
        # If you want to visualize “contested” packages distinctly, do it here.

        sim.curr_drone_site_locs = CurrDroneSiteLocs(curr_drone_locs_cols, curr_sites_locs_cols)


    else:
        for drone_nm, v in list(server.agent_task_allocation.items()):
            package_nm, _ = v
            flag = "None"

            delivery, return_time = sim.true_delivery_return[(drone_nm, package_nm)]

            logging.info(f"Drone {drone_nm} has package {package_nm} with true delivery {delivery} and return {return_time}, the window is {server.agent_task_windows[(drone_nm, package_nm)]}")

            # triggering only if deliveries are happening in this time step
            # For pickup, assign package to drone and mark package inactive
            if (delivery > old_time) and (delivery <= sim.current_time):
                server.agent_prop_set[drone_nm].current_package = ""
                server.agent_prop_set[drone_nm].at_depot = False  # For good measure


                # print("sim", sim.busy_packages[pkg])

                # Check on-time vs late
                if delivery <= sim.busy_packages[package_nm].time_window[1]:
                    sim.delivered_packages += 1
                    logging.info(f"{drone_nm} has delivered {package_nm}!")
                    # Drone is green and at package location
                    flag = "True"
                    curr_drone_locs_cols.append((sim.busy_packages[package_nm].delivery, "green"))
                else:
                    sim.late_packages += 1
                    logging.info(f"{package_nm} was not delivered in time!")
                    # Drone is red and at package location
                    flag = "False"
                    curr_drone_locs_cols.append((sim.busy_packages[package_nm].delivery, "red"))
                
                if csv_logger:
                    # pkg_window = server.agent_task_windows[(dn, pkg)]
                    csv_logger.log("final_deliveries.csv", {
                        "trial": getattr(sim, "trial_id", None),
                        "timestep": sim.current_time,
                        "drone_id": drone_nm,
                        "pkg_id": package_nm,
                        "actual_delivery_time": delivery,
                        "deadline": sim.busy_packages[package_nm].time_window[1],
                        "on_time": flag,
                    })

                busy_packages_to_del.add(package_nm)
                sim.done_packages[package_nm] = sim.busy_packages[package_nm]

                # Increment loss with difference from start of window
                sim.sum_of_delivery_time += delivery - server.agent_task_windows[(drone_nm, package_nm)][0]

                logging.debug(
                    f"[Delivery] t={sim.current_time} | Drone {drone_nm} delivered pkg {package_nm} "
                    f"at {delivery:.2f}, deadline={sim.busy_packages[package_nm].time_window[1]:.2f}, "
                    f"on_time={delivery <= sim.busy_packages[package_nm].time_window[1]}"
                )
            # For dropoff, free up drone and setup deletion of (drone,package) keys
            # Return to depot occurred here
            elif (return_time > old_time) and (return_time <= sim.current_time):
                server.agent_prop_set[drone_nm].at_depot = True
                server.agent_prop_set[drone_nm].available_at = return_time  
                server.agent_prop_set[drone_nm].current_package = ""  
                keys_to_del.add((drone_nm, package_nm))
                logging.info(f"{drone_nm} back at depot!")
                done_packages_to_del.add(package_nm)

                # Drone is blue and at depot
                curr_drone_locs_cols.append((server.agent_set[drone_nm].depot_loc, "blue"))

            else:
                # NOTE: Hacky plotting interpolation
                # if pkg in sim.busy_packages or pkg in sim.done_packages:
                if (package_nm not in sim.busy_packages) and (package_nm not in sim.done_packages):
                    continue
                depot_loc = server.agent_set[drone_nm].depot_loc
                
                if sim.current_time < delivery:
                    # Interpolate depot -> delivery
                    pp_loc = sim.busy_packages[package_nm].delivery
                    maxdiff = max(delivery, (return_time - delivery))
                    interp_factor = (delivery - sim.current_time) / maxdiff if maxdiff != 0 else 0.0
                    new_lat = depot_loc.lat + (1.0 - interp_factor) * (pp_loc.lat - depot_loc.lat)
                    new_lon = depot_loc.lon + (1.0 - interp_factor) * (pp_loc.lon - depot_loc.lon)
                else:
                    # Interpolate delivery -> depot
                    pp_loc = sim.done_packages[package_nm].delivery
                    denom = (return_time - delivery)
                    interp_factor = (return_time - sim.current_time) / denom if denom != 0 else 0.0
                    new_lat = depot_loc.lat + (interp_factor) * (pp_loc.lat - depot_loc.lat)
                    new_lon = depot_loc.lon + (interp_factor) * (pp_loc.lon - depot_loc.lon)

                curr_drone_locs_cols.append((type(depot_loc)(lat=new_lat, lon=new_lon), "blue"))


        # Delete bookkeeping keys for dropped-off package
        for k in keys_to_del:
            sim.true_delivery_return.pop(k, None)
            server.agent_task_windows.pop(k, None)
            # Only delete the agent key from allocation
            server.agent_task_allocation.pop(k[0], None)

        for r in busy_packages_to_del:
            sim.busy_packages.pop(r, None)

        for r in done_packages_to_del:
            sim.done_packages.pop(r, None)

        # 3) Handle active packages that expired without attempt
        packages_to_del = set()
        for package_nm, rp in list(sim.active_packages.items()):
            if rp.time_window[1] <= sim.current_time:
                logging.info(f"{package_nm} was not even attempted!")
                sim.late_packages += 1
                packages_to_del.add(package_nm)
                # Add package location with red
                curr_sites_locs_cols.append((rp.delivery, "red"))

        for r in packages_to_del:
            sim.active_packages.pop(r, None)
            sim.num_active_packages -= 1

        # # 4) Generate new packages probabilistically
        # if rng.random() <= sim.new_request_prob:
        #     lat_start, lat_end = sim.city_params.lat_start, sim.city_params.lat_end
        #     lon_start, lon_end = sim.city_params.lon_start, sim.city_params.lon_end
        #     new_package = generate_package_request(
        #         # Sample uniformly in bounds
        #         rng.uniform(lat_start, lat_end),
        #         rng.uniform(lon_start, lon_end),
        #         sim.current_time,
        #         sim.time_window_duration,
        #         rng
        #     )
        #     sim.num_total_packages += 1
        #     sim.num_active_packages += 1
        #     new_package_nm = f"pkg{sim.num_total_packages}"
        #     logging.info(f"{new_package_nm} added!")
        #     sim.active_packages[new_package_nm] = new_package

        # 5) Grey markers for sites
        for pkg_nm, pp in sim.active_packages.items():
            curr_sites_locs_cols.append((pp.delivery, "grey"))
        for pkg_nm, pp in sim.busy_packages.items():
            curr_sites_locs_cols.append((pp.delivery, "grey"))

        sim.curr_drone_site_locs = CurrDroneSiteLocs(curr_drone_locs_cols, curr_sites_locs_cols)



def update_time_windows(sim, server, csv_logger=None) -> None:
    """
    Recompute interaction events and time windows for all active packages.

    - Ensures sim and server times match.
    - For each drone and each active package, if that (drone, package) pair
      doesn't yet have a time-window entry, compute/record:
        * the drone's return_time estimate (delivery -> depot),
        * the interaction timestamps {START, FINISH, RETURN},
        * the server-side agent_task_windows triple [start, finish, return].
    - Finally, sorts each drone's interaction_events by RETURN time.
    """
    assert sim.current_time == server.current_time, "sim and server times must match"

    for drone_nm, drone_props in server.agent_prop_set.items():
        interaction_events = drone_props.interaction_events
        drone = server.agent_set[drone_nm]
        drone_depot = drone.depot_number


        # For every active package, initialize per-(drone,package) timing if missing
        for package_nm, rp in sim.active_packages.items():
            key = (drone_nm, package_nm)
            if key not in server.agent_task_windows:
                 # travel time to return to depot after delivery
                mean_travel_time = rp.approx_travel_times.get(drone_depot)

                # Build interaction timestamps
                start_t = rp.time_window[0]
                finish_t = rp.time_window[1]
                return2depot_t = math.ceil(finish_t) + mean_travel_time

                ie_timestamps = {
                    MODE.START:  start_t,
                    MODE.FINISH: finish_t,
                    MODE.RETURN: return2depot_t,
                }

                # Record event + server-side window triple
                interaction_events.append(InteractionEvent(drone_nm, package_nm, ie_timestamps, mean_travel_time))
                server.agent_task_windows[key] = [start_t, finish_t, return2depot_t]
                csv_logger.log("interaction_events.csv", {
                    "timestep": sim.current_time,
                    "drone": drone_nm,
                    "package": package_nm,
                    "earliest_time_attempt": start_t,
                    "latest_time_attempt": finish_t,
                    "success_return": return2depot_t,
                })

        # now sort by the return timestamp
        interaction_events.sort(key=lambda ev: ev.timestamps[MODE.FINISH])
        server.agent_prop_set[drone_nm].interaction_events = interaction_events



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
