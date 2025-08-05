# scoba_conflict_resolution.py
import heapq
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Callable, Any
from solver.scoba_tree_search import generate_search_tree, get_next_attempt_idx
TaskUtil = Tuple[str, float]

@dataclass(order=True)
class SCoBAHighLevelNode:
    """
    High-level node for SC0BA conflict resolution.
    Ordering is by utility (max-heap uses negative utility).
    """
    util: float
    id: int = field(compare=False)
    task_allocation: Dict[str, TaskUtil] = field(compare=False, default_factory=dict)
    considered_tasks: Dict[str, Set[str]] = field(compare=False, default_factory=dict)
    constraints: Dict[str, Set[str]] = field(compare=False, default_factory=dict)

class SCoBAAlgorithm:
    """
    Handles constraint-based coordination of task allocations.
    """
    def __init__(self, allocation: Any, routing_sim: Any):
        self.allocation = allocation
        self.routing_sim = routing_sim
        # internal heap stores tuples: (-util, id, node)
        self.heap: list[Tuple[float, int, SCoBAHighLevelNode]] = []
        self.num_total_conflicts: int = 0

    def coordinate_allocation(
        self,
        initial_task_allocation: Dict[str, TaskUtil],
        initial_considered_tasks: Dict[str, Set[str]],
        init_util: float,
        success_prob_fn: Callable,
        util_val_fn: Callable,
        downtime: float
    ) -> Dict[str, TaskUtil]:
        """
        Resolve conflicts in an initial allocation by generating alternative assignments.
        Returns a conflict-free task_allocation mapping.
        """
        # Create root node
        start = SCoBAHighLevelNode(
            util=init_util,
            id=0,
            task_allocation=initial_task_allocation.copy(),
            considered_tasks=deepcopy(initial_considered_tasks),
            constraints={k: set() for k in initial_considered_tasks.keys()}  
        )
        heapq.heappush(self.heap, (-start.util, start.id, start))
        node_id = 1

        while self.heap:
            _, _, P = heapq.heappop(self.heap)
            # Build map from task -> agents assigned
            task_to_agents: Dict[str, list[str]] = {}
            for agent, tu in P.task_allocation.items():
                task = tu[0]
                task_to_agents.setdefault(task, []).append(agent)

            num_conflicts = 0
            # tasks_to_ignore = set() 
            tasks_to_ignore = set(task_to_agents.keys())
            # print(f"[Coord] Tasks to ignore: {tasks_to_ignore}")

            # For each task with multiple agents, branch
            conflict_tasks = [task for task, agents in task_to_agents.items() if len(agents) > 1]
            num_conflicts = len(conflict_tasks)
            logging.info(f"[Coord] Total conflicts this round: {num_conflicts}")
            for task, agents in task_to_agents.items():
                if len(agents) > 1:
                    for agt in agents:
                        logging.debug(f"[Coord] Conflict on task '{task}' assigned to agents: {agents}")
                        new_node = deepcopy(P)
                        new_node.id = node_id
                        # Agents other than the one we allow
                        other_agents = set(agents) - {agt}
                        for other in other_agents:
                            # Remove the problematic allocation
                            removed_util = new_node.task_allocation[other][1]
                            new_node.util -= removed_util
                            del new_node.task_allocation[other]
                            # Exclude tasks globally
                            # new_node.considered_tasks[other] -= tasks_to_ignore
                            # new_node.considered_tasks[other].discard(task)
                            before = new_node.considered_tasks[other].copy()
                            new_node.considered_tasks[other] -= tasks_to_ignore
                            after = new_node.considered_tasks[other]
                            removed = before - after
                            logging.debug(f"[Coord] Actually removed from agent {other}'s list: {removed}")
                            # if other not in new_node.considered_tasks:
                            #     new_node.considered_tasks[other] = set()
                            # new_node.considered_tasks[other].discard(task)
                            # Recompute low-level tree for this agent
                            generate_search_tree(
                                self.allocation,
                                other,
                                new_node.considered_tasks[other],
                                success_prob_fn,
                                util_val_fn,
                                downtime
                            )
                            tree = self.allocation.agent_prop_set[other].tree
                            dec_idx = get_next_attempt_idx(tree)
                            if dec_idx != -1:
                                dec_node = tree.nodes[dec_idx]
                                new_node.util += dec_node.util
                                new_node.task_allocation[other] = (dec_node.task_name, dec_node.util)
                        heapq.heappush(self.heap, (-new_node.util, new_node.id, new_node))
                        # logging.debug(f"[Coord] Candidate node {new_node.id} assigns {len(new_node.task_allocation)} drones with total util={new_node.util:.2f}")
                        # logging.debug(f"[Coord] Added node {new_node.id} with constraint {task} on agent {other}")


                        node_id += 1
                    num_conflicts += 1
                    self.num_total_conflicts += 1
                    # logging.debug(f"[Coord] Conflict count {self.num_total_conflicts} exceeds threshold {getattr(self.allocation, 'conflict_threshold', float('inf'))}")

                    # Bail out if too many conflicts
                    if self.num_total_conflicts > getattr(self.allocation, 'conflict_threshold', float('inf')):
                        logging.warning("High level conflict threshold exceeded!")
                        # return P.task_allocation
                        return {}
            # If no conflicts, we have a valid allocation
            if num_conflicts == 0:
                logging.debug("No more conflicts; returning allocation.")
                return P.task_allocation

        logging.info("No coordinated allocation found; returning empty allocation.")
        return {}
