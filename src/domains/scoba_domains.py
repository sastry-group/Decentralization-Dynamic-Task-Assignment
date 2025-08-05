import os
import sys
# Ensure project root is on path so `solver` package is importable
CUR_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# core routing types
from domains.routing.routing_types import (
    LatLonCoords,
    convert_to_vector,
    EuclideanLatLongMetric,
    CityParams,
    parse_city_params,
    Drone,
    DroneProperties,
    Package,
    CurrDroneSiteLocs,
    RoutingSimulator,
)
# allocation type
from solver.scoba_types import GenericAllocation as RoutingAllocation

# MCTS code (moved into mcts.py)
from domains.routing.mcts import (
    RoutingMCTSMDP,
    update_routing_mcts_fullstate,
    update_server_with_mdp_action,
    NearestPkgPolicy,
)

# baseline heuristics
from domains.routing.routing_baselines import (
    expected_hungarian,
    earliest_due_date,
    get_current_routing_mcts_state,
    RoutingDroneState,
)

# scoba dispatching baseline
from domains.routing.routing_scoba import (
    delivery_util,
    delivery_success_prob,
    scoba_routing,
)

# simulator utilities
from domains.routing.routing_simulator import (
    get_travel_time_estimate,
    generate_package_request,
    sample_true_delivery_return_time,
    setup_routing_sim,
    update_routing_sim,
    update_time_windows,
    parse_routing_commandline,
)

__all__ = [
    # Core types
    "LatLonCoords", "convert_to_vector", "EuclideanLatLongMetric",
    "CityParams", "parse_city_params",
    "Drone", "DroneProperties", "Package", "CurrDroneSiteLocs", "RoutingSimulator",
    # Allocation
    "RoutingAllocation",
    # Baselines & MCTS
    "expected_hungarian", "earliest_due_date", "RoutingDroneState",
    "RoutingMCTSMDP", "update_routing_mcts_fullstate",
    "get_current_routing_mcts_state", "update_server_with_mdp_action",
    "NearestPkgPolicy",
    # SC0BA
    "delivery_util", "delivery_success_prob", "scoba_routing",
    # Simulator utilities
    "get_travel_time_estimate", "generate_package_request",
    "sample_true_delivery_return_time", "setup_routing_sim",
    "update_routing_sim", "update_time_windows", "parse_routing_commandline",
]
