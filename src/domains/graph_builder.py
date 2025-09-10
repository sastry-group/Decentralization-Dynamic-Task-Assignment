from typing import Dict, List, Tuple

def build_comm_structure(depots: Dict[int, list], comms_dict: dict = None):
    """
    Build a mapping of which depots can see which other depots.
    comms_dict format: {depot_id: [list of visible depot_ids]}.
    If comms_dict is None, default to full communication (all-to-all).
    """
    depot_ids = list(depots.keys())
    if comms_dict is None:
        # Full graph: each depot sees all others (including itself)
        return {d: depot_ids for d in depot_ids}
    else:
        # Ensure each depot exists in the dictionary
        return {d: comms_dict.get(d, []) for d in depot_ids}