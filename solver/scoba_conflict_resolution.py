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
           


class SCoBAAlgorithmDecentralized:

    def __init__(self, allocation: Any, routing_sim: Any, 
                 comm_graph: dict[str, set[str]],
                 undirected: bool = True,
                 is_task_visible: Callable[[str, str], bool] | None = None):
        """
        comm_graph: adjacency list over agent ids (depots). Edge => can coordinate/share info.
        undirected: if True, treat conflicts only when there's a reciprocal link. If False, allow directed neighborhoods.
        is_task_visible(agent, task): optional predicate to restrict what tasks an agent can even consider (e.g., depot reach).
        """
        self.allocation = allocation
        self.routing_sim = routing_sim
        
        self.comm_graph = {a: set(neigh) for a, neigh in comm_graph.items()}
        if undirected:
            for a, neigh in list(self.comm_graph.items()):
                for b in list(neigh):
                    self.comm_graph.setdefault(b, set()).add(a)
        self.is_task_visible = is_task_visible or (lambda a, t: True)
        self.heap: list[Tuple[float, int, SCoBAHighLevelNode]] = []
        self.num_total_conflicts: int = 0

    def _components(self) -> list[set[str]]:
        seen, comps = set(), []
        for a in self.comm_graph:
            if a in seen: 
                continue
            stack, comp = [a], set()
            while stack:
                u = stack.pop()
                if u in seen: 
                    continue
                seen.add(u); comp.add(u)
                stack.extend(self.comm_graph.get(u, ()))
            comps.append(comp)
        return comps
    

    def coordinate_allocation_components(
        self, initial_task_allocation, initial_considered_tasks, init_util,
        success_prob_fn, util_val_fn, downtime) -> dict[str, TaskUtil]:
        final = {}
        for comp in self._components():
            # slice inputs to the component
            alloc_c = {a: tu for a, tu in initial_task_allocation.items() if a in comp}
            cons_c  = {a: {t for t in initial_considered_tasks.get(a, set()) 
                        if self.is_task_visible(a, t)} for a in comp}
            # run the regular high-level, but it will only see comp agents
            sub = self.coordinate_allocation(
                alloc_c, cons_c, init_util=init_util,
                success_prob_fn=success_prob_fn, util_val_fn=util_val_fn, downtime=downtime,
                agent_filter=comp
            )
            final.update(sub)
        return final
    
    def coordinate_allocation(
        self,
        initial_task_allocation: Dict[str, TaskUtil],
        initial_considered_tasks: Dict[str, Set[str]],
        init_util: float,
        success_prob_fn: Callable,
        util_val_fn: Callable,
        downtime: float,
        agent_filter: set[str] | None = None) -> Dict[str, TaskUtil]:
        agents_scope = agent_filter or set(initial_considered_tasks.keys())

        start = SCoBAHighLevelNode(
            util=init_util,
            id=0,
            task_allocation={a: tu for a, tu in initial_task_allocation.items() if a in agents_scope},
            considered_tasks={k: {t for t in v if self.is_task_visible(k, t)}
                            for k, v in initial_considered_tasks.items() if k in agents_scope},
            constraints={k: set() for k in agents_scope}
        )
        self.heap.clear()
        heapq.heappush(self.heap, (-start.util, start.id, start))
        node_id = 1

        while self.heap:
            _, _, P = heapq.heappop(self.heap)

            # Task -> agents assigned (restricted to scope)
            task_to_agents: Dict[str, list[str]] = {}
            for agent, tu in P.task_allocation.items():
                if agent not in agents_scope: 
                    continue
                task = tu[0]
                task_to_agents.setdefault(task, []).append(agent)

            # Build conflicts but ONLY among agents that have an edge (adjacent in comm graph)
            conflict_pairs: list[tuple[str, str, str]] = []  # (task, a, b)
            for task, agents in task_to_agents.items():
                if len(agents) < 2:
                    continue
                # consider the induced subgraph and only flag pairs with an edge
                for i in range(len(agents)):
                    for j in range(i+1, len(agents)):
                        a, b = agents[i], agents[j]
                        # conflict only if they can coordinate (share info)
                        if (b in self.comm_graph.get(a, set())) or (a in self.comm_graph.get(b, set())):
                            conflict_pairs.append((task, a, b))

            if not conflict_pairs:
                # no local conflicts under comm graph; return as "locally valid"
                return P.task_allocation

            # Branch on each local conflict
            for (task, a, b) in conflict_pairs:
                for keep in (a, b):
                    new_node = deepcopy(P); new_node.id = node_id
                    drop = b if keep == a else a

                    # remove drop's current allocation and decrease util
                    if drop in new_node.task_allocation:
                        removed_util = new_node.task_allocation[drop][1]
                        new_node.util -= removed_util
                        del new_node.task_allocation[drop]

                    # prune drop's considered tasks ONLY by:
                    #   (i) keeping visibility
                    #  (ii) *optional* removing the specific task that caused a conflict
                    before = new_node.considered_tasks.get(drop, set()).copy()
                    pruned = {t for t in before if self.is_task_visible(drop, t)}
                    pruned.discard(task)   # local exclusion, not global blanket removes
                    new_node.considered_tasks[drop] = pruned

                    # recompute the low-level plan for 'drop' using only its visible tasks and local context
                    generate_search_tree(
                        self.allocation, drop, new_node.considered_tasks[drop],
                        success_prob_fn, util_val_fn, downtime
                    )
                    tree = self.allocation.agent_prop_set[drop].tree
                    dec_idx = get_next_attempt_idx(tree)
                    if dec_idx != -1:
                        dec_node = tree.nodes[dec_idx]
                        new_node.util += dec_node.util
                        new_node.task_allocation[drop] = (dec_node.task_name, dec_node.util)

                    heapq.heappush(self.heap, (-new_node.util, new_node.id, new_node))
                    node_id += 1

            self.num_total_conflicts += 1
            if self.num_total_conflicts > getattr(self.allocation, 'conflict_threshold', float('inf')):
                return {}
        return {}
