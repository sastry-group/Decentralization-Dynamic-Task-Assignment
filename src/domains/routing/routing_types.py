import math
import sys
from dataclasses import dataclass, field
from typing import NamedTuple, Tuple, Dict, Any, List
import numpy as np

# Placeholder imports:
# from sklearn.neighbors import BallTree

class LatLonCoords(NamedTuple):
    lat: float
    lon: float

Location2D = np.ndarray  # alias for vector form [lat, lon]

def convert_to_vector(c: LatLonCoords) -> Location2D:
    """Return [lat, lon] as a numpy array."""
    return np.array([c.lat, c.lon])

class EuclideanLatLongMetric:
    """Compute approximate Euclidean distance on lat/lon coordinates."""
    def evaluate(self, coords1: Location2D, coords2: Location2D) -> float:
        deglen = 110.25
        x = coords1[0] - coords2[0]
        # adjust longitude by latitude
        y = (coords1[1] - coords2[1]) * math.cos(math.radians(coords2[0]))
        return deglen * math.hypot(x, y)

@dataclass
class CityParams:
    lat_start: float
    lat_end: float
    lon_start: float
    lon_end: float



def parse_city_params(param_file: str) -> CityParams:
    """Read LATSTART/LATEND/LONSTART/LONEND from a TOML file."""
    if sys.version_info >= (3, 11):
        import tomllib
        params = tomllib.load(open(param_file, 'rb'))
    else:
        import toml
        params = toml.load(param_file)
    return CityParams(
        lat_start=params["LATSTART"],
        lon_start=params["LONSTART"],
        lat_end=params["LATEND"],
        lon_end=params["LONEND"]
    )

@dataclass
class Depot:
    depot_id: str
    location: LatLonCoords
    capacity: int = 5  # not used in the current implementation


@dataclass
class Drone:
    drone_id: str
    avg_speed: float = 0.00777 # km/s, approx. 28 km/h
    depot_number: int = 0
    depot_loc: LatLonCoords = LatLonCoords(0.0, 0.0)

@dataclass
class DroneProperties:
    at_depot: bool = True
    current_package: str = ""
    interaction_events: List[Any] = field(default_factory=list)
    tree: Any = field(default_factory=lambda: None)  # placeholder for SearchTree

@dataclass
class Package:
    delivery: LatLonCoords
    time_window: Tuple[float, float]
    approx_travel_times: Dict[int, float] = field(default_factory=dict)

@dataclass
class CurrDroneSiteLocs:
    curr_drone_locs_cols: List[Tuple[LatLonCoords, str]] = field(default_factory=list)
    curr_sites_locs_cols: List[Tuple[LatLonCoords, str]] = field(default_factory=list)

@dataclass
class RoutingSimulator:
    current_time: float
    city_params: CityParams
    new_request_prob: float
    time_window_duration: float
    halton_nn_tree: Any  # BallTree
    estimate_matrix: np.ndarray
    delivery_reward: float
    active_packages: Dict[str, Package] = field(default_factory=dict)
    busy_packages: Dict[str, Package]   = field(default_factory=dict)
    done_packages: Dict[str, Package]   = field(default_factory=dict)
    in_transit_packages: Dict[str, Package] = field(default_factory=dict)
    num_total_packages: int             = 0
    num_active_packages: int            = 0
    late_packages: int                  = 0
    delivered_packages: int             = 0
    sum_of_delivery_time: float         = 0.0
    time_scale: float                   = 60.0  #  I think to transfor from minutes to seconds
    true_delivery_return: Dict[Tuple[str, str], Tuple[float, float]] = field(default_factory=dict)
    tt_est_std_scale: float             = 3.0
    distance_thresh: float              = 5.0 # kilometers
    curr_drone_site_locs: CurrDroneSiteLocs = field(default_factory=CurrDroneSiteLocs)
    depot_locs: Dict[str, LatLonCoords] = field(default_factory=dict)


