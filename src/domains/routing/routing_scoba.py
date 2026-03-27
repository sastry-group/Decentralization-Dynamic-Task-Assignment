import numpy as np
from typing import Dict, List, Any, Set, Tuple
from collections import namedtuple
import logging
import random

# Solver types
from solver.scoba_tree_search import generate_search_tree, get_next_attempt_idx
from solver.scoba_conflict_resolution import SCoBAAlgorithm

from domains.routing.routing_types import EuclideanLatLongMetric, convert_to_vector
from domains.routing.travel_model import delivery_success_prob, sample_true_delivery_return_time


TaskUtil = namedtuple("TaskUtil", ["task", "util"])


def make_success_prob_fn(std_scale, drone_nm):
    def success_prob(ref_time, ie):
        return delivery_success_prob(ref_time=ref_time, ie=ie, std_scale=std_scale)
    return success_prob



def scoba_routing(server, routing_sim, rng: Any = None, csv_logger=None, trial_id=None, time_step=None,
                  comms_dict=None, allow_overlap=False, depot_order=None) -> None:
    """
    Assign tasks to drones using SCoBA with UNDIRECTED communication groups.

    Interpretation:
      - comms_dict defines depot-level communication/visibility
      - for SCoBA we treat it as UNDIRECTED
      - each connected component of available depots is solved jointly
      - full comms  -> one component -> one joint SCoBA solve
      - no comms    -> singleton components
      - partial     -> one SCoBA solve per connected component
    """
    def util_val_fn(_):
        return routing_sim.delivery_reward

    def success_prob_factory(drone_nm: str):
        return make_success_prob_fn(routing_sim.tt_est_std_scale, drone_nm)


    def build_undirected_comms(comms_dict_in, depots: Set[int]) -> Dict[int, Set[int]]:
        """
        Symmetrize depot comms:
        if i sees j OR j sees i, connect i--j.
        """
        undirected = {d: set() for d in depots}
        if comms_dict_in is None:
            for d in depots:
                undirected[d] = set(depots - {d})
            return undirected

        for d in depots:
            for nd in comms_dict_in.get(d, []):
                if nd in depots and nd != d:
                    undirected[d].add(nd)
                    undirected[nd].add(d)
        return undirected

    def connected_components_undirected(graph: Dict[int, Set[int]], nodes: Set[int]) -> List[Set[int]]:
        visited = set()
        components = []

        for start in sorted(nodes):
            if start in visited:
                continue
            stack = [start]
            comp = set()
            visited.add(start)

            while stack:
                u = stack.pop()
                comp.add(u)
                for v in graph.get(u, set()):
                    if v in nodes and v not in visited:
                        visited.add(v)
                        stack.append(v)

            components.append(comp)

        return components

    # ------------------------------------------------------------------
    # 1. Group available drones by depot
    # ------------------------------------------------------------------
    available_drones: Dict[int, List[str]] = {}
    all_drones_by_depot: Dict[int, List[str]] = {}

    for drone_nm, dp in server.agent_prop_set.items():
        dnum = server.agent_set[drone_nm].depot_number
        all_drones_by_depot.setdefault(dnum, []).append(drone_nm)
        if dp.at_depot:
            # n_events = len(server.agent_prop_set[drone_nm].interaction_events)
            # logging.info(f"[SCoBA] drone={drone_nm} interaction_events={n_events} "
            #          f"tree_nodes={len(server.agent_prop_set[drone_nm].tree.nodes)}")
            available_drones.setdefault(dnum, []).append(drone_nm)

    if not available_drones:
        logging.info(f"[SCoBA] t={time_step} no drones at depot; skipping.")
        return

    available_depots = set(available_drones.keys())


    # ------------------------------------------------------------------
    # 2. Build UNDIRECTED depot comms graph and connected components
    # ------------------------------------------------------------------
    undirected_comms = build_undirected_comms(comms_dict, available_depots)
    components = connected_components_undirected(undirected_comms, available_depots)

    # optional component ordering
    if depot_order == "desc":
        components = sorted(components, key=lambda c: max(c), reverse=True)
    elif depot_order == "random":
        components = random.sample(components, len(components))
    else:
        components = sorted(components, key=lambda c: min(c))

    # logging.info(f"[SCoBA] available_depots={sorted(available_depots)}")
    # logging.info(
    #     "[SCoBA] undirected_comms=" +
    #     str({d: sorted(list(neigh)) for d, neigh in undirected_comms.items()})
    # )
    # logging.info(
    #     "[SCoBA] components=" +
    #     str([sorted(list(c)) for c in components])
    # )

    # ------------------------------------------------------------------
    # 3. Compute previously-visible packages PER DEPOT using undirected neighborhoods
    # ------------------------------------------------------------------
    previously_visible: Dict[int, Set[str]] = {d: set() for d in all_drones_by_depot}

    if allow_overlap:
        active_now = set(routing_sim.active_packages.keys())

        # logging.info(
        #     f"[SCoBA] active_pkgs={sorted(routing_sim.active_packages.keys())}"
        # )
        # logging.info(
        #     f"[SCoBA] package_claims=" +
        #     str({pkg: sorted(list(claims)) for pkg, claims in getattr(routing_sim, 'package_claims', {}).items()})
        # )
        # logging.info(
        #     f"[SCoBA] package_winners={getattr(routing_sim, 'package_winners', {})}"
        # )
        for d in all_drones_by_depot:
            vis_depots = {d} | undirected_comms.get(d, set())

            vis_drones = set()
            for vd in vis_depots:
                vis_drones.update(all_drones_by_depot.get(vd, []))

            prev: Set[str] = set()

            for pkg, claimants in getattr(routing_sim, "package_claims", {}).items():
                if pkg in active_now and (set(claimants) & vis_drones):
                    prev.add(pkg)

            for pkg, winner_dn in getattr(routing_sim, "package_winners", {}).items():
                if pkg in active_now and winner_dn in vis_drones:
                    prev.add(pkg)

            previously_visible[d] = prev

        # for d in sorted(previously_visible):
        #     logging.info(f"[SCoBA] depot {d} previously-visible pkgs: {sorted(previously_visible[d])}")

    # ------------------------------------------------------------------
    # 3. Determine depot processing order
    # ------------------------------------------------------------------
    # if depot_order == "desc":
    #     ordered_depots = sorted(available_drones.keys(), reverse=True)
    # elif depot_order == "random":
    #     ordered_depots = random.sample(list(available_drones.keys()), len(available_drones))
    # else:
    #     ordered_depots = sorted(available_drones.keys())

    # ------------------------------------------------------------------
    # 4. Build tentative allocations PER COMPONENT, not per global depot loop
    # ------------------------------------------------------------------
    scoba_alg = SCoBAAlgorithm(allocation=server, routing_sim=routing_sim)
    globally_assigned: Set[str] = set() 

    for component in components:
        component_depots = sorted(component)
        component_drones = []
        for d in component_depots:
            component_drones.extend(available_drones.get(d, []))

        # logging.info(
        #     f"[SCoBA][component] depots={component_depots} "
        #     f"drones={component_drones}"
        # )

        if not component_drones:
            continue

        task_util_allocation: Dict[str, TaskUtil] = {}
        all_considered_tasks: Dict[str, Set[str]] = {}
        assignment_util = 0.0

        if depot_order == "desc":
            ordered_component_depots = sorted(component_depots, reverse=True)
        elif depot_order == "random":
            ordered_component_depots = random.sample(component_depots, len(component_depots))
        else:
            ordered_component_depots = sorted(component_depots)

        for depot_number in ordered_component_depots:
            depot_drones = available_drones.get(depot_number, [])
            if not depot_drones:
                continue

            depot_loc = server.agent_set[depot_drones[0]].depot_loc

            pkgs_in_range: Set[str] = {
                pkg_nm for pkg_nm, pp in routing_sim.active_packages.items()
                if EuclideanLatLongMetric().evaluate(
                    convert_to_vector(depot_loc), convert_to_vector(pp.delivery)
                ) <= routing_sim.distance_thresh
            }

            if allow_overlap:
                pkgs_available = pkgs_in_range - previously_visible.get(depot_number, set()) - globally_assigned  
            else:
                pkgs_available = pkgs_in_range - set(routing_sim.busy_packages.keys()) - globally_assigned 

            depot_assigned_pkgs: Set[str] = set()

            for drone_nm in depot_drones:
                # logging.info(f"[SCoBA] drone={drone_nm} PRE-build "
                            # f"interaction_events={len(server.agent_prop_set[drone_nm].interaction_events)} "
                            # f"tree_nodes={len(server.agent_prop_set[drone_nm].tree.nodes)}")
                pkgs_to_consider = pkgs_available - depot_assigned_pkgs
                all_considered_tasks[drone_nm] = set(pkgs_to_consider)

                if not pkgs_to_consider:
                    continue

                sp_fn = success_prob_factory(drone_nm)
                generate_search_tree(server, drone_nm, pkgs_to_consider, sp_fn, util_val_fn, 0.0)
                tree = server.agent_prop_set[drone_nm].tree
                # logging.info(f"[SCoBA] drone={drone_nm} post-build "
                #             f"interaction_events={len(server.agent_prop_set[drone_nm].interaction_events)} "
                #             f"tree_nodes={len(tree.nodes)}")
                if not tree:
                    continue

                dec_idx = get_next_attempt_idx(tree)
                if dec_idx == -1:
                    continue

                dec_node = tree.nodes[dec_idx]
                depot_assigned_pkgs.add(dec_node.task_name)
                assignment_util += dec_node.util
                task_util_allocation[drone_nm] = TaskUtil(task=dec_node.task_name, util=dec_node.util)

                # logging.info(
                #     f"[SCoBA][tentative] drone={drone_nm} depot={depot_number} "
                #     f"chosen={dec_node.task_name} util={dec_node.util:.4f}"
                # )

        # logging.info(
        #     "[SCoBA][pre-conflict] " +
        #     str({dn: {"task": tu.task, "util": round(tu.util, 4)}
        #          for dn, tu in task_util_allocation.items()})
        # ) 

        if not task_util_allocation:
            continue

        # --------------------------------------------------------------
        # Deconflict inside this component only
        # --------------------------------------------------------------
        if len(component) == 1:
            group_allocation = task_util_allocation
        else:
            considered_in_group = {
                dn: {pkg for pkg in all_considered_tasks.get(dn, set())
                     if pkg in routing_sim.active_packages}
                for dn in task_util_allocation
            }

            util_in_group = sum(tu.util for tu in task_util_allocation.values())

            group_allocation = scoba_alg.coordinate_allocation(
                task_util_allocation,
                considered_in_group,
                util_in_group,
                success_prob_factory,
                util_val_fn,
                0.0,
            )

        # logging.info(
        #     "[SCoBA][post-conflict] " +
        #     str({dn: {"task": tu.task, "util": round(tu.util, 4)}
        #          for dn, tu in group_allocation.items()})
        # )

        # --------------------------------------------------------------
        # Commit immediately for this component
        # --------------------------------------------------------------
        for drone_nm, pkg_util in group_allocation.items():
            pkg_nm = pkg_util.task
            depot_number = server.agent_set[drone_nm].depot_number
            depot_loc = server.agent_set[drone_nm].depot_loc

            if drone_nm in server.agent_task_allocation:
                continue
            if pkg_nm not in routing_sim.active_packages:
                continue
            # if (not allow_overlap) and (pkg_nm in routing_sim.busy_packages):
            #     continue
            if pkg_nm in globally_assigned:   
                continue

            globally_assigned.add(pkg_nm)  
            window = routing_sim.active_packages[pkg_nm].time_window
            td, rt = sample_true_delivery_return_time(
                depot_loc,
                routing_sim.active_packages[pkg_nm].delivery,
                window,
                server.current_time,
                rng,
            )

            server.agent_task_allocation[drone_nm] = (pkg_nm, float("inf"))
            routing_sim.true_delivery_return[(drone_nm, pkg_nm)] = (td, rt)
            server.agent_prop_set[drone_nm].at_depot = False
            server.agent_prop_set[drone_nm].current_package = pkg_nm
            routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages[pkg_nm]

            if allow_overlap:
                routing_sim.package_claims.setdefault(pkg_nm, set()).add(drone_nm)
                routing_sim.package_registry[pkg_nm]["claimed_by"].append(drone_nm)
                routing_sim.package_registry[pkg_nm]["time_assigned"].append(time_step)
            else:
                routing_sim.busy_packages[pkg_nm] = routing_sim.active_packages.pop(pkg_nm)
                routing_sim.num_active_packages -= 1




            if csv_logger:
                csv_logger.log("drone_assignment.csv", {
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
                    "true_travel_time": rt - td,
                })