# scoba_types.py

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

class MODE(IntEnum):
    START   = 1
    FINISH  = 2
    SUCCESS = 3

@dataclass
class InteractionEvent:
    agent_name: str
    task_name: str
    timestamps: Dict[MODE, float]
    travel_time: float = 0.0

@dataclass
class DecisionNode:
    agent_name: str
    task_name: str
    attempt:   bool
    timeval:   float
    util:      float = float("-inf")
    idx:       int   = 0
    utilset:   bool  = False

@dataclass
class OutcomeNode:
    agent_name:  str
    task_name:   str
    outcome:     MODE
    timeval:     float
    probability: float
    util:        float = float("-inf")
    idx:         int   = 0
    utilset:     bool  = False

class SearchTree:
    def __init__(self):
        self.nodes:      List[DecisionNode|OutcomeNode] = []
        self.child_ids:  Dict[int, List[int]]          = {}
        self.parent_id:  Dict[int, int]                = {}

@dataclass
class GenericAllocation:
    current_time: float                    = 0.0
    agent_set: Dict[str, Any]              = field(default_factory=dict)
    agent_prop_set: Dict[str, Any]         = field(default_factory=dict)
    agent_task_windows: Dict[Tuple[str,str], Tuple[float,float,float]] = field(default_factory=dict) # start of time window, end of time window for delivery, Nominal travel time / expected delivery time
    agent_task_allocation: Dict[str, Tuple[str,float]]               = field(default_factory=dict)
    agent_ordering: List[str]              = field(default_factory=list)
    max_tasks_to_consider: int             = 100_000
    conflict_threshold: int                = 100


# mark it as the common base class for your sim
class MRTAEnvironment: pass

# and
TaskUtil = Tuple[str, float]
