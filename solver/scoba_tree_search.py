# scoba_tree_search.py
from collections import deque
import math
from typing import Any
from solver.scoba_types import DecisionNode, OutcomeNode, SearchTree, InteractionEvent, MODE
import logging



def ensure_root(tree):
    if not tree.nodes:
        tree.nodes.append(DecisionNode(agent_name="", task_name="", attempt=False,
                                       timeval=0.0, util=0.0, idx=0, utilset=True))
        tree.parent_id[0] = 0
        tree.child_ids[0] = []

def insert_decision_node(tree: SearchTree, agent_name: str, ie: InteractionEvent,
                         parent_id: int, ref_time: float, utilval: float,
                         success_prob_fn) -> tuple[list[int], int]:
    """
    Insert a new decision for interaction event `ie` under `parent_id`.

    Returns:
        (new_outcome_leaves, new_decision_leaf_idx)
    """

    # Convenience: append node and return its new index (assumes tree.nodes[0] is dummy/root)
    def _append(node):
        idx = len(tree.nodes)
        tree.nodes.append(node)
        return idx
    

    # 1) True reference time is clamped to the event START
    START, FINISH, RETURN = MODE.START, MODE.FINISH, MODE.RETURN
    true_ref_time = max(ref_time, ie.timestamps[MODE.START])
    assert true_ref_time <= ie.timestamps[MODE.FINISH], f"ref_time={ref_time} exceeds FINISH; ie={ie.timestamps}"

    # 2) Create the two decision nodes (attempt / no-attempt), util initialized to 0
    attempt_idx = _append(DecisionNode(
        agent_name=agent_name,
        task_name=ie.task_name,
        attempt=True,
        timeval=true_ref_time,
        util=0.0,
        idx=None,           # will set below if your class stores idx internally
        utilset=False
    ))
    tree.parent_id[attempt_idx] = parent_id

    # Decision node for no-attempt
    no_attempt_idx = _append(DecisionNode(
        agent_name=agent_name,
        task_name=ie.task_name,
        attempt=False,
        timeval=true_ref_time,
        util=0.0,
        idx=None,
        utilset=False
    ))
    tree.parent_id[no_attempt_idx] = parent_id
    new_decision_leaf = no_attempt_idx

    # If nodes store their own id, set it now
    if hasattr(tree.nodes[attempt_idx], "idx"):
        tree.nodes[attempt_idx].idx = attempt_idx
    if hasattr(tree.nodes[no_attempt_idx], "idx"):
        tree.nodes[no_attempt_idx].idx = no_attempt_idx

    # Parent now has two children: [attempt, no-attempt]
    tree.child_ids[parent_id] = [attempt_idx, no_attempt_idx]

    # 3) Success probability from your callback (Julia uses ref_time, not true_ref_time)
    success_prob = float(success_prob_fn(ref_time, ie))
    success_prob = max(0.0, min(1.0, success_prob))  # clamp, just in case

    # 4) Create outcome nodes under the "attempt" decision
    fail_idx = _append(OutcomeNode(
        agent_name=agent_name,
        task_name=ie.task_name,
        outcome=FINISH,  # failure realized at FINISH
        timeval=ie.timestamps[FINISH],
        probability=1.0 - success_prob,
        util=0.0,        # leaf util gets set later in the backward pass
        idx=None,
        utilset=False
    ))
    tree.parent_id[fail_idx] = attempt_idx

    succ_idx = _append(OutcomeNode(
        agent_name=agent_name,
        task_name=ie.task_name,
        outcome=RETURN,  # success realized at RETURN
        timeval=ie.timestamps[RETURN],
        probability=success_prob,
        util=utilval,    # immediate reward for success
        idx=None,
        utilset=False
    ))
    tree.parent_id[succ_idx] = attempt_idx

    if hasattr(tree.nodes[fail_idx], "idx"):
        tree.nodes[fail_idx].idx = fail_idx
    if hasattr(tree.nodes[succ_idx], "idx"):
        tree.nodes[succ_idx].idx = succ_idx

    # Children of the attempt decision are [success, fail]
    tree.child_ids[attempt_idx] = [succ_idx, fail_idx]

    new_outcome_leaves = [succ_idx, fail_idx]
    return new_outcome_leaves, new_decision_leaf


def generate_search_tree(server: Any, agent_name: str, tasks_to_consider: set[str],
                         success_prob_fn, util_val_fn, downtime: float) -> None:
    """
    Build or rebuild the search tree for a single agent based on tasks to consider.
    """
    if not tasks_to_consider:
        temp_tree = server.agent_prop_set[agent_name].tree
        temp_tree.nodes.clear()
        temp_tree.child_ids.clear()
        temp_tree.parent_id.clear()
        return
    

    tree = server.agent_prop_set[agent_name].tree
    interaction_events = server.agent_prop_set[agent_name].interaction_events

    outcome_leaf_idxs = set()
    decision_leaf_idxs = set()

    first_ie_stamp = math.inf

    # Filter to packages in tasks_to_consider; keep original order or sort by START
    considered_ie = [ie for ie in interaction_events if ie.task_name in tasks_to_consider]
    num_to_consider = min(len(considered_ie), server.max_tasks_to_consider)
    logging.info(f"[SCoBA] Agent {agent_name} considering {num_to_consider} tasks out of {len(considered_ie)}")
    if num_to_consider == 0:
        tree.clear()
        return
    
    # Optional: be explicit about ordering by START (safe)
    considered_ie = sorted(considered_ie[:num_to_consider], key=lambda ie: ie.timestamps[MODE.START])

    for ie in considered_ie:
        ie_stamp = ie.timestamps[MODE.START]


        if ie_stamp < first_ie_stamp:
            
            utilval = util_val_fn(ie)
            new_outcomes, new_dec_leaf = insert_decision_node(
                tree, agent_name, ie, 0, server.current_time, utilval, success_prob_fn
            )
            for idx in new_outcomes:
                outcome_leaf_idxs.add(idx)
            decision_leaf_idxs.add(new_dec_leaf)
            first_ie_stamp = ie_stamp
        else:
            new_outcome_leaf_idxs = set()
            new_decision_leaf_idxs = set()
            outcome_leaves_to_rm = set()
            decision_leaves_to_rm = set()

            # Try inserting under current outcome leaves (non‑dominated checks)
            for ol_idx in list(outcome_leaf_idxs):
                onode = tree.nodes[ol_idx]
                assert type(onode).__name__ == "OutcomeNode"

                # If the outcome happens before this IE starts -> dominated, skip
                if onode.timeval <= ie.timestamps[MODE.START]:
                    continue
                # If this decision would finish too late (beyond FINISH - downtime/2), skip
                if onode.timeval >= ie.timestamps[MODE.FINISH] - downtime/2.0:
                    continue

                # Non-dominated: add decision node below this outcome leaf
                utilval = util_val_fn(ie)
                temp_outcomes, temp_dec = insert_decision_node(
                    tree, agent_name, ie, ol_idx, onode.timeval, utilval, success_prob_fn
                )
                outcome_leaves_to_rm.add(ol_idx)
                for to_idx in temp_outcomes:
                    new_outcome_leaf_idxs.add(to_idx)
                new_decision_leaf_idxs.add(temp_dec)

            # Not-attempt is different from failing immediately; ensure paired growth
            assert (len(new_decision_leaf_idxs) == 0) == (len(new_outcome_leaf_idxs) == 0)                

            # Insert under current "no-attempt" decision leaves
            for dec_idx in list(decision_leaf_idxs):
                dnode = tree.nodes[dec_idx]
                assert getattr(dnode, "attempt") is False
                if dnode.timeval >= ie.timestamps[MODE.FINISH] - downtime/2.0:
                    continue

                utilval = util_val_fn(ie)
                temp_outcomes, temp_dec = insert_decision_node(
                    tree, agent_name, ie, dec_idx, dnode.timeval, utilval, success_prob_fn
                )
                decision_leaves_to_rm.add(dec_idx)
                for to_idx in temp_outcomes:
                    new_outcome_leaf_idxs.add(to_idx)
                new_decision_leaf_idxs.add(temp_dec)

            # Update leaf sets
            outcome_leaf_idxs -= outcome_leaves_to_rm
            outcome_leaf_idxs |= new_outcome_leaf_idxs

            decision_leaf_idxs -= decision_leaves_to_rm
            decision_leaf_idxs |= new_decision_leaf_idxs


    # Backward pass: set utilities on leaves then bubble up to root
    rev_fringe_idxs = set()

    # Outcome leaves: mark utilset (their .util already set by insert_decision_node normally)
    for ol_idx in outcome_leaf_idxs:
        ol_node = tree.nodes[ol_idx]
        ol_node.utilset = True
        rev_fringe_idxs.add(ol_idx)

    # Decision "no-attempt" leaves: util = 0
    for dec_idx in decision_leaf_idxs:
        dec_node = tree.nodes[dec_idx]
        assert getattr(dec_node, "attempt") is False
        dec_node.util = 0.0
        dec_node.utilset = True
        rev_fringe_idxs.add(dec_idx)

    # Bubble up to set parent utilities
    while rev_fringe_idxs:
        to_rm = set()
        to_add = set()

        for rfi in list(rev_fringe_idxs):
            # Stop at root's children (parent_id == 0), remove from fringe
            if tree.parent_id[rfi] == 0:
                to_rm.add(rfi)
                continue

            node = tree.nodes[rfi]
            assert node.utilset is True, "Node in rev fringe does NOT have util set"

            par_id = tree.parent_id[rfi]
            par_node = tree.nodes[par_id]

            # Only update parent when *all* its children are in the fringe (i.e., utilset ready)
            siblings = tree.child_ids[par_id]
            if not all(child in rev_fringe_idxs for child in siblings):
                continue

            # Update parent util depending on child/parent types
            if type(node).__name__ == "OutcomeNode":
                # Parent is a decision node here: add expected utility
                par_node.util = par_node.util + node.probability * node.util
            else:
                # node is a DecisionNode
                # If parent is outcome SUCCESS, add +1 then take best; else just take best
                tmp_util = node.util
                if type(par_node).__name__ == "OutcomeNode" and par_node.outcome == MODE.RETURN:
                    tmp_util = 1.0 + node.util
                par_node.util = max(tmp_util, getattr(par_node, "util", float("-inf")))

            to_add.add(par_id)
            to_rm.add(rfi)

        for pid in to_add:
            tree.nodes[pid].utilset = True

        rev_fringe_idxs |= to_add
        rev_fringe_idxs -= to_rm
    

def get_next_attempt_idx(tree: 'SearchTree') -> int:
    """
    Traverse decision tree preferring higher utility branches.
    Return the first valid attempt=True decision node. 
    Exhausts queue if necessary.
    """
    if not tree.nodes:
        return -1

    # Start BFS from the root (assumed index 0)
    dec_fringe = deque([0])
    subtree_root_dec = -1

    while dec_fringe:
        dec_top = dec_fringe.popleft()

        children = tree.child_ids.get(dec_top, [])
        # assert len(top_children) == 2, "Each decision node must have exactly two children"


        if len(children) != 2:
            # no children yet → keep scanning others
            continue

        c0, c1 = children
        n0, n1 = tree.nodes[c0], tree.nodes[c1]

        def is_attempt(node): return hasattr(node, "attempt") and node.attempt is True
        def is_dec(node):     return hasattr(node, "attempt")

        if is_dec(n0) and is_dec(n1):
            attempt_idx  = c0 if is_attempt(n0) else c1
            no_attempt_idx = c1 if attempt_idx == c0 else c0

            # If skipping yields higher util, keep going down that branch; else choose attempt here
            if tree.nodes[no_attempt_idx].util > tree.nodes[attempt_idx].util:
                dec_fringe.append(no_attempt_idx)
            else:
                subtree_root_dec = attempt_idx
                # don’t descend further from here (mirrors Julia logic)
        else:
            # Children aren’t both decisions (likely outcomes) → nothing to do here
            continue

    return subtree_root_dec
