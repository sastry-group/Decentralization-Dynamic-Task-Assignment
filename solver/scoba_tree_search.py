# scoba_tree_search.py
from collections import deque
from typing import Any
from solver.scoba_types import DecisionNode, OutcomeNode, SearchTree, InteractionEvent, MODE
import logging


def insert_decision_node(tree: SearchTree, agent_name: str, ie: InteractionEvent,
                         parent_id: int, ref_time: float, utilval: float,
                         success_prob_fn) -> tuple[list[int], int]:
    """
    Insert decision and outcome nodes into the search tree for a given interaction event.
    Returns (new_outcome_leaves, new_decision_leaf).
    """
    idx = len(tree.nodes)
    # Clamp reference time to event start
    true_ref_time = max(ref_time, ie.timestamps[MODE.START])
    assert true_ref_time <= ie.timestamps[MODE.FINISH], f"ref_time {ref_time} beyond event {ie.timestamps}"

    # Decision node for attempt
    attempt_idx = idx
    dnode1 = DecisionNode(agent_name=agent_name,
                           task_name=ie.task_name,
                           attempt=True,
                           timeval=true_ref_time,
                           util=0.0,
                           idx=attempt_idx,
                           utilset=False)
    tree.nodes.append(dnode1)
    tree.parent_id[attempt_idx] = parent_id

    # Decision node for no-attempt
    no_attempt_idx = attempt_idx + 1
    dnode2 = DecisionNode(agent_name=agent_name,
                           task_name=ie.task_name,
                           attempt=False,
                           timeval=true_ref_time,
                           util=0.0,
                           idx=no_attempt_idx,
                           utilset=False)
    tree.nodes.append(dnode2)
    tree.parent_id[no_attempt_idx] = parent_id
    tree.child_ids[parent_id] = [attempt_idx, no_attempt_idx]

    # Outcome nodes under attempt
    success_prob = success_prob_fn(ref_time, ie)
    fail_idx = no_attempt_idx + 1
    onode1 = OutcomeNode(agent_name=agent_name,
                         task_name=ie.task_name,
                         outcome=MODE.FINISH,
                         timeval=ie.timestamps[MODE.FINISH],
                         probability=1.0 - success_prob,
                         util=0.0,
                         idx=fail_idx,
                         utilset=False)
    tree.nodes.append(onode1)
    tree.parent_id[fail_idx] = attempt_idx

    succ_idx = fail_idx + 1
    onode2 = OutcomeNode(agent_name=agent_name,
                         task_name=ie.task_name,
                         outcome=MODE.RETURN,
                         timeval=ie.timestamps[MODE.RETURN],
                         probability=success_prob,
                         util=utilval,
                         idx=succ_idx,
                         utilset=False)
    tree.nodes.append(onode2)
    tree.parent_id[succ_idx] = attempt_idx
    tree.child_ids[attempt_idx] = [succ_idx, fail_idx]
    # logging.info(
    #     f"[Outcome Added] Task {ie.task_name} — success_prob={success_prob:.4f}, "
    #     f"utilval={utilval:.2f}, succ_time={ie.timestamps[MODE.SUCCESS]:.2f}, "
    #     f"fail_time={ie.timestamps[MODE.FINISH]:.2f}, ref_time={ref_time:.2f}"
    # )

    return [succ_idx, fail_idx], no_attempt_idx


def generate_search_tree(server: Any, agent_name: str, tasks_to_consider: set[str],
                         success_prob_fn, util_val_fn, downtime: float) -> None:
    """
    Build or rebuild the search tree for a single agent based on tasks to consider.
    """
    prop = server.agent_prop_set[agent_name]
    tree = prop.tree
    if not tasks_to_consider:
        tree.nodes.clear()
        tree.parent_id.clear()
        tree.child_ids.clear()
        return

    events = prop.interaction_events
    considered_ie = [ie for ie in events if ie.task_name in tasks_to_consider]
    task_summary = {} 
    max_consider = min(len(considered_ie), server.max_tasks_to_consider)
    outcome_leaf_idxs: set[int] = set()
    decision_leaf_idxs: set[int] = set()
    first_time = float('inf')

    logging.debug(f"[SCoBA] Drone {agent_name} has {len(considered_ie)} interaction events. Using max {server.max_tasks_to_consider}")

    for ie in considered_ie[:max_consider]:
        ie_stamp = ie.timestamps[MODE.START]
        # logging.info(f"[RewardCheck] Task {ie.task_name} reward={util_val_fn(ie):.2f}")

        if ie_stamp < first_time:
            # to initialize the tree
            utilval = util_val_fn(ie)
            logging.info(f"[UtilCheck] Task first time {ie.task_name} → util={util_val_fn(ie):.2f}")
            new_outs, new_dec = insert_decision_node(
                tree, agent_name, ie, 0, server.current_time, utilval, success_prob_fn
            )
            outcome_leaf_idxs.update(new_outs)
            decision_leaf_idxs.add(new_dec)
            first_time = ie_stamp
        else:
            new_outs_all = set()
            new_decs_all = set()
            remove_outs = set()
            remove_decs = set()
            for ol in list(outcome_leaf_idxs):
                onode = tree.nodes[ol]
                if onode.timeval <= ie.timestamps[MODE.START] or onode.timeval >= ie.timestamps[MODE.FINISH] - downtime/2.0:
                    continue
                utilval = util_val_fn(ie)
                # logging.info(f"[UtilCheck] Task {ie.task_name} → util={util_val_fn(ie):.2f}")
                temp_outs, temp_dec = insert_decision_node(
                    tree, agent_name, ie, ol, onode.timeval, utilval, success_prob_fn
                )
                remove_outs.add(ol)
                new_outs_all.update(temp_outs)
                new_decs_all.add(temp_dec)
            for dl in list(decision_leaf_idxs):
                dnode = tree.nodes[dl]
                if dnode.timeval < ie.timestamps[MODE.FINISH] - downtime/2.0:
                    utilval = util_val_fn(ie)
                    temp_outs, temp_dec = insert_decision_node(
                        tree, agent_name, ie, dl, dnode.timeval, utilval, success_prob_fn
                    )
                    remove_decs.add(dl)
                    new_outs_all.update(temp_outs)
                    new_decs_all.add(temp_dec)
            outcome_leaf_idxs.difference_update(remove_outs)
            outcome_leaf_idxs.update(new_outs_all)
            decision_leaf_idxs.difference_update(remove_decs)
            decision_leaf_idxs.update(new_decs_all)
        task_summary[ie.task_name] = utilval


    
    # if task_summary:
    #     log_lines = [f"[SCoBA Summary] Drone {agent_name} considered {len(task_summary)} tasks:"]
    #     for task, util in task_summary.items():
    #         log_lines.append(f"  - Task {task}: utility={util:.2f}")
    #     logging.info("\n".join(log_lines))
    # else:
    #     logging.info(f"[SCoBA Summary] Drone {agent_name} had no tasks to consider.")
    # excluded = {ie.task_name for ie in events} - tasks_to_consider
    # if excluded:
    #     logging.info(f"[SCoBA Summary] Drone {agent_name} ignored {len(excluded)} tasks: {sorted(excluded)}")

    # for node in tree.nodes:
    #     if isinstance(node, OutcomeNode):
    #         logging.info(f"[Debug Outcome] Task {node.task_name} — util={node.util}, prob={node.probability}, timeval={node.timeval}")

    reverse_fringe = set(outcome_leaf_idxs) | set(decision_leaf_idxs)
    while reverse_fringe:
        to_remove = set()
        to_add = set()
        for node_idx in list(reverse_fringe):
            parent = tree.parent_id.get(node_idx, None)
            if parent is None or parent == 0:
                to_remove.add(node_idx)
                continue
            siblings = tree.child_ids[parent]
            if not all(s in reverse_fringe for s in siblings):
                continue
            node = tree.nodes[node_idx]
            par_node = tree.nodes[parent]
            if isinstance(node, OutcomeNode):
                # logging.info(f"[Backup] Outcome for {node.task_name}, prob={node.probability:.4f}, util={node.util:.2f}")
                par_node.util += node.probability * node.util
            else:
                temp = node.util + (1.0 if isinstance(par_node, OutcomeNode) and par_node.outcome == MODE.RETURN else 0.0)
                par_node.util = max(par_node.util, temp)
            to_add.add(parent)
            to_remove.add(node_idx)
        reverse_fringe.difference_update(to_remove)
        reverse_fringe.update(to_add)
        for idx in to_add:
            tree.nodes[idx].utilset = True
    
    # f
def get_next_attempt_idx(tree: 'SearchTree') -> int:
    """
    Traverse decision tree preferring higher utility branches.
    Return the first valid attempt=True decision node. 
    Exhausts queue if necessary.
    """
    if not tree.nodes:
        return -1

    dq = deque([0])  # root node index
    while dq:
        top = dq.popleft()
        children = tree.child_ids.get(top, [])
        decs = [c for c in children if hasattr(tree.nodes[c], "attempt")]
        if len(decs) != 2:
            continue

        attempt_idx = next(c for c in decs if tree.nodes[c].attempt)
        no_attempt_idx = next(c for c in decs if not tree.nodes[c].attempt)

        if tree.nodes[no_attempt_idx].util > tree.nodes[attempt_idx].util:
            dq.append(no_attempt_idx)
        else:
            return attempt_idx

    return -1