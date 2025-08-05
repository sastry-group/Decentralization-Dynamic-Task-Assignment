
import sys

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:
    import toml as _toml 

def parse_city_params(toml_path: str):
    """
    Reads a TOML with keys
      LATSTART, LONSTART, LATEND, LONEND
    and returns a dict with float values.
    """
    with open(toml_path, "rb") as f:
        params = _toml.load(f)
    return {
        "lat_start": params["LATSTART"],
        "lon_start": params["LONSTART"],
        "lat_end":   params["LATEND"],
        "lon_end":   params["LONEND"],
    }