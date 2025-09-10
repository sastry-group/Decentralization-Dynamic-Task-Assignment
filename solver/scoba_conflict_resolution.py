# scoba_conflict_resolution.py
from __future__ import annotations
import heapq
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Callable, Any, NamedTuple, Optional
from solver.scoba_tree_search import generate_search_tree, get_next_attempt_idx

SuccessProbCB = Callable[[float, Any], float]                 # (ref_time, ie) -> prob
SuccessProbFactory = Callable[[str], SuccessProbCB]          # drone_nm -> callback

class TaskUtil(NamedTuple):
    task: str
    util: float

@dataclass(order=True)
class SCoBAHighLevelNode:
    """
    High-level node for SC0BA conflict resolution.
    Ordering is by utility (max-heap uses negative utility).
    """
    util: float
    id: int = field(compare=False, default=0)
    task_allocation: Dict[str, TaskUtil] = field(compare=False, default_factory=dict)  # agent -> (task, util)
    considered_tasks: Dict[str, Set[str]] = field(compare=False, default_factory=dict) # agent -> set(tasks)
    # constraints: Dict[str, Set[str]] = field(compare=False, default_factory=dict)

class SCoBAAlgorithm:
    """
    Handles constraint-based coordination of task allocations.
    """
    def __init__(self, allocation, routing_sim=None):
        self.allocation = allocation
        self.routing_sim = routing_sim
        self.heap = []          # real list
        self.num_total_conflicts = 0

    def coordinate_allocation(
            self,
            initial_task_allocation: Dict[str, TaskUtil],
            initial_considered_tasks: Dict[str, Set[str]],
            init_util: float,
            success_prob_factory: SuccessProbFactory, 
            util_val_fn: Callable[[Any], float],
            downtime: float) -> Dict[str, TaskUtil]:

        # Create root node
        start = SCoBAHighLevelNode(
            util=init_util,
            id=0,
            task_allocation=initial_task_allocation.copy(),
            considered_tasks=copy.deepcopy(initial_considered_tasks),
            # constraints={k: set() for k in initial_considered_tasks.keys()}
        )
        self.heap.clear()
        heapq.heappush(self.heap, (-start.util, start.id, start))
        next_id = 1

        while self.heap:
            _, _, P = heapq.heappop(self.heap)

            # Build reverse map: task -> [agents who got it]
            task_to_agents: Dict[str, list[str]] = {}
            for agent, tu in P.task_allocation.items():
                t = tu.task
                task_to_agents.setdefault(t, []).append(agent)

            # Count conflicts and prepare to branch
            num_conflicts = 0
            tasks_to_ignore_global = set(task_to_agents.keys())


            for task, agents in task_to_agents.items():
                if len(agents) <= 1:
                    continue  # no conflict for this task


                num_conflicts += 1
                self.num_total_conflicts += 1

                # For each agent in the conflict, spawn a child that forbids this task for all *other* conflicted agents
                for keep_agent in agents:
                    new_node = copy.deepcopy(P)
                    new_node.id = next_id
                    next_id += 1
                    # Other conflicted agents to modify
                    others = set(agents)
                    others.discard(keep_agent)

                    for other in others:
                        # Remove previous utility contribution and its allocation for that agent
                        if other in new_node.task_allocation:
                            new_node.util -= new_node.task_allocation[other].util
                            del new_node.task_allocation[other]

                        # Forbid the entire batch of current tasks (same as Julia: remove all tasks currently considered)
                        # Then re-run low-level search on the remaining allowed tasks.
                        # In Julia they did: setdiff!(considered_tasks[other_agt], tasks_to_ignore)
                        if other in new_node.considered_tasks:
                            new_node.considered_tasks[other].difference_update(tasks_to_ignore_global)

                        sp_other = success_prob_factory(other)
                        # Recompute the low-level tree for 'other' agent
                        generate_search_tree(
                            self.allocation,
                            other,
                            new_node.considered_tasks.get(other, set()),
                            sp_other,
                            util_val_fn,
                            downtime,
                        )                        

                        tree = self.allocation.agent_prop_set[other].tree
                        dec_idx = get_next_attempt_idx(tree)

                        if dec_idx != -1:
                            dec_node = tree.nodes[dec_idx]
                            # Update HL node util and allocation for 'other'
                            new_node.util += float(dec_node.util)
                            new_node.task_allocation[other] = TaskUtil(task=str(dec_node.task_name),
                                                                    util=float(dec_node.util))

                    # Push child into heap
                    heapq.heappush(self.heap, (-new_node.util, new_node.id, new_node))

                if self.num_total_conflicts > self.allocation.conflict_threshold:
                    # Return the best-so-far (current popped node P) allocation

                    logging.warning(f"Exceeded conflict threshold: {self.num_total_conflicts} > {self.allocation.conflict_threshold}")
                    return P.task_allocation
                
            # If no conflicts, we’re done
            if num_conflicts == 0:
                return P.task_allocation

        # If heap exhausts, return empty allocation
        return {}
           