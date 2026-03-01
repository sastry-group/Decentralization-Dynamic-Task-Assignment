import numpy as np
import math
from typing import Tuple
from solver.scoba_types import InteractionEvent
from domains.routing.routing_types import EuclideanLatLongMetric, convert_to_vector
from .routing_types import LatLonCoords, Package, CurrDroneSiteLocs, CityParams, parse_city_params
from solver.scoba_types import MODE


TRAVEL = dict(
    avg_speed_km_per_min = 0.00777 * 60 / 1.2, # ~0.4662 km/min , just a scale factor to icnrease travel times
    cv = 0.33,           # stdev = cv * mean   (tune 0.2–0.4 to taste)
    dist = "epanechnikov"  # "epanechnikov" or "normal"
)

def travel_time_mean_minutes(loc1, loc2) -> float:
    v1, v2 = convert_to_vector(loc1), convert_to_vector(loc2)
    dist_km = EuclideanLatLongMetric().evaluate(v1, v2)
    mu = dist_km / TRAVEL["avg_speed_km_per_min"]
    return max(math.ceil(mu), 3)


def _epanechnikov_cdf_u(u: float) -> float:
    """
    CDF of standardized Epanechnikov with support [-1,1].
    """
    if u <= -1.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return 0.5 + 0.75 * (u - (u**3) / 3.0)

def epanechnikov_cdf(x: float, mean: float, sigma: float) -> float:
    """
    Epanechnikov CDF with mean and standard deviation sigma.
    """
    if sigma <= 0:
        return 1.0 if x >= mean else 0.0

    u = (x - mean) / sigma
    return _epanechnikov_cdf_u(u)


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
    
def delivery_success_prob(ref_time: float, ie: InteractionEvent, std_scale: float) -> float:
    mu = ie.travel_time
    sigma = mu / std_scale
    x = ie.timestamps[MODE.FINISH] - ref_time
    return epanechnikov_cdf(x, mu, sigma)


def sample_true_travel_time(mu: float, rng: np.random.Generator) -> float:
    sigma = max(TRAVEL["cv"] * mu, 1e-6)
    if TRAVEL["dist"] == "epanechnikov":
        # u ~ Epanechnikov with Var(u)=1 using the sqrt(5) trick; std = sigma
        sqrt5 = 5 ** 0.5
        while True:
            u = rng.uniform(-sqrt5, sqrt5)
            if rng.uniform() <= 0.75 * (1 - (u / sqrt5) ** 2):
                return max(1.0, round(mu + u * sigma))
    elif TRAVEL["dist"] == "normal":
        return max(1.0, round(rng.normal(mu, sigma)))
    else:
        raise ValueError("Unknown TRAVEL['dist']")
    

def sample_true_delivery_return_time(
    depot_loc: LatLonCoords,
    delivery_loc: LatLonCoords,
    window: Tuple[float, float],
    current_time: float,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """
    Sample true delivery time (arrive at customer) and true return time (back to depot),
    using the same uncertainty model as everywhere else.
    """
    mu_out = travel_time_mean_minutes(depot_loc, delivery_loc)
    mu_back = travel_time_mean_minutes(delivery_loc, depot_loc)

    # sample both legs with the same distribution family & CV
    tt_out = sample_true_travel_time(mu_out, rng)
    tt_back = sample_true_travel_time(mu_back, rng)

    # depart immediately; arrive at
    td = round(current_time + tt_out)
    # respect time window start: wait if early
    td = max(td, window[0])

    # return after (waiting does not reduce flight time)
    rt = round(td + tt_back)

    return td, rt

# def delivery_success_prob(std_scale: float, ref_time: float, ie: InteractionEvent) -> float:
#     travel_time = ie.travel_time # some estimate  trvel time
#     mean = travel_time
#     scale = travel_time / std_scale
#     x = ie.timestamps[MODE.FINISH] - ref_time  # this is counting what the current time step is 

#     prob = epanechnikov_cdf(x, mean, scale)
#     return prob


