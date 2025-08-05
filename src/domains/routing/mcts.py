# mcts.py
from copy import deepcopy
import numpy as np
from pomdp_py.framework.basics import State, Action, GenerativeDistribution, BlackboxModel, PolicyModel
from pomdp_py.algorithms.po_uct import RolloutPolicy
from pomdp_py import Agent
from pomdp_py.algorithms.po_uct import POUCT
from typing import List, Any
from .routing_simulator import (
    sample_true_delivery_return_time,
    update_time_windows,
    update_routing_sim,
)

from .routing_types import convert_to_vector, EuclideanLatLongMetric

class RoutingAction(Action):
    def __init__(self, pkg_id: int):
        self.pkg_id = pkg_id
    def __hash__(self):
        return hash(self.pkg_id)
    def __eq__(self, other):
        return isinstance(other, RoutingAction) and self.pkg_id == other.pkg_id
    def __repr__(self):
        return f"RoutingAction({self.pkg_id})"


class RoutingAgent(Agent):
    def __init__(self, mdp):
        b0 = PointMass(mdp.state)
        bb = RoutingBlackBox(mdp)
        pm = RoutingPolicyModel(mdp) 
        super().__init__(
            init_belief       = b0,
            policy_model      = pm,          
            transition_model  = None,
            observation_model = None,
            reward_model      = None,
            blackbox_model    = bb
        )
        self.mdp = mdp



class RoutingState(State):
    __slots__ = ("sim","server")
    def __init__(self, sim, server):
        self.sim, self.server = sim, server
    def __hash__(self):
        return hash((self.server.current_time,
                     frozenset(self.server.agent_task_allocation.items())))
    def __eq__(self, other):
        return (isinstance(other, RoutingState)
                and self.server.current_time == other.server.current_time
                and self.server.agent_task_allocation == other.server.agent_task_allocation)

class RoutingMCTSMDP:
    def __init__(self, sim, server, horizon:int):
        self.routing_sim, self.server, self.horizon = sim, server, horizon
        self.drone_pkg_assignment = []
        self.true_delivery_return = []
    @property
    def state(self)->RoutingState:
        return RoutingState(deepcopy(self.routing_sim),
                            deepcopy(self.server))
    def copy(self): 
        return RoutingMCTSMDP(deepcopy(self.routing_sim),
                              deepcopy(self.server),
                              self.horizon)
    def step(self, state: RoutingState, joint_action: List[RoutingAction]):
        """
        joint_action: one RoutingAction per drone (in the same order as server.agent_ordering)
        """
        # clone
        sim_snap = deepcopy(state.sim)
        srv_snap = deepcopy(state.server)
        tmp = RoutingMCTSMDP(sim_snap, srv_snap, self.horizon)
        # apply the whole vector of actions, then advance one step
        from .mcts import update_server_with_mdp_action
        update_server_with_mdp_action(tmp, joint_action)
        return tmp.state
    

    def reward(self, s, a, s2):    return 0.0
    def observe(self, s, a, s2):   return s2
    def discount(self):            return 1.0
    def is_terminal(self, s)->bool: return s.server.current_time >= self.horizon

    def _apply_action(self, sim, server, action: RoutingAction):
        # 1) Decode action → pkg name
        pkg = f"pkg{action.pkg_id}"
        # 2) Figure out which drone this action is for. 
        #    (If you're doing joint actions, this method should take the full action-vector;
        #     otherwise, keep track of a single-drone context.)
        dn = ...  

        # 3) Assign the package
        server.agent_task_allocation[dn] = (pkg, float('inf'))
        td = sample_true_delivery_return_time(
            server.agent_task_windows[(dn, pkg)],
            server.current_time,
            sim.tt_est_std_scale,
            rng=None
        )
        sim.true_delivery_return[(dn, pkg)] = td
        server.agent_prop_set[dn].at_depot = False
        sim.busy_packages[pkg] = sim.active_packages.pop(pkg)
        sim.num_active_packages -= 1

        # 4) Advance one timestep
        update_time_windows(sim, server)
        update_routing_sim(sim, server, rng=None)

        return sim, server



class RoutingBlackBox(BlackboxModel):
    def __init__(self, mdp):      self._mdp = mdp
    def sample(self, state, action):
        m = self._mdp.copy()
        ns = m.step(state, action)
        r  = m.reward(state, action, ns)
        return ns, ns, r
    def argmax(self, state, action):
        return self.sample(state, action)
    

class PointMass(GenerativeDistribution):
    def __init__(self, state):
        self._state = state
    def mpe(self):
        return self._state
    def random(self):
        return self._state
    def __getitem__(self, s):
        return 1.0 if s == self._state else 0.0

class RoutingPolicyModel(PolicyModel):
    def __init__(self, mdp): super().__init__(); self.mdp=mdp
    def get_all_actions(self, state, history=None):
        active = list(self.mdp.routing_sim.active_packages)
        ids     = [int(p.replace("pkg","")) for p in active]
        return [RoutingAction(0)] + [RoutingAction(i) for i in ids]
    def probability(self, a, s, history=None):
        A=self.get_all_actions(s); return 1/len(A) if A else 0.0
    def sample(self, s, history=None):
        A=self.get_all_actions(s); return A[np.random.randint(len(A))] if A else RoutingAction(0)
    def argmax(self, s, history=None):
        A=self.get_all_actions(s); return A[0] if A else RoutingAction(0)

class NearestPkgPolicy(RolloutPolicy):
    def __init__(self, mdp): super().__init__(); self.mdp=mdp
    def sample(self, state):
        # same “nearest‐pkg” logic but wrap in RoutingAction
        if state.pkg_assignment!=0: return RoutingAction(0)
        assigned={a.pkg_id for a in self.mdp.drone_pkg_assignment if a>0}
        dn = self.mdp.server.agent_ordering[state.drone_idx-1]
        depot = self.mdp.server.agent_set[dn].depot_loc
        best,bd=0,self.mdp.routing_sim.distance_thresh
        for pkg,pp in self.mdp.routing_sim.active_packages.items():
            pid=int(pkg.replace("pkg",""))
            if pid in assigned: continue
            dist=EuclideanLatLongMetric().evaluate(
                   convert_to_vector(depot), convert_to_vector(pp.delivery))
            if dist<=bd: bd, best = dist,pid
        return RoutingAction(best)
    



def update_routing_mcts_fullstate(mdp: RoutingMCTSMDP, rng: Any = None) -> None:
    """See Julia update_routing_mcts_fullstate!"""
    dp_asg: List[int] = []
    tr_list: List[Tuple[float, float]] = []
    for dn in mdp.server.agent_ordering:
        dp = mdp.server.agent_prop_set[dn]
        if dp.current_package:
            p = int(dp.current_package.replace("pkg", ""))
            td = sample_true_delivery_return_time(
                mdp.server.agent_task_windows[(dn, dp.current_package)],
                mdp.server.current_time,
                mdp.routing_sim.tt_est_std_scale,
                rng,
            )
            dp_asg.append(p)
            tr_list.append(td)
        else:
            if dp.at_depot:
                dp_asg.append(0)
                tr_list.append((0.0, 0.0))
            else:
                dp_asg.append(-1)
                for (k, (pkg, _)) in mdp.server.agent_task_allocation.items():
                    if k == dn:
                        td = sample_true_delivery_return_time(
                            mdp.server.agent_task_windows[(dn, pkg)],
                            mdp.server.current_time,
                            mdp.routing_sim.tt_est_std_scale,
                            rng,
                        )
                        tr_list.append(td)
                        break
    mdp.drone_pkg_assignment = dp_asg
    mdp.true_delivery_return = tr_list


def update_server_with_mdp_action(
    mdp: RoutingMCTSMDP,
    actions: List[RoutingAction],
    rng: np.random.Generator = None
) -> None:
    """
    Given your joint action vector (one RoutingAction per drone), mutate
    the MDP’s `.server` and `.routing_sim` just as in your benchmark loop:
    assign packages, sample delivery/return times, then tick the world one step.
    """
    if rng is None:
        from .routing_simulator import sample_true_delivery_return_time
        from .routing_simulator import update_time_windows, update_routing_sim
        import numpy as _np
        rng = _np.random.default_rng()

    # 1) decode and apply each drone’s action
    for act, dn in zip(actions, mdp.server.agent_ordering):
        pkg = f"pkg{act.pkg_id}"
        # skip “no‐op”:
        if act.pkg_id == 0 or pkg not in mdp.routing_sim.active_packages:
            continue
        # schedule the assignment
        mdp.server.agent_task_allocation[dn] = (pkg, float("inf"))
        td, rt = sample_true_delivery_return_time(
            mdp.server.agent_task_windows[(dn, pkg)],
            mdp.server.current_time,
            mdp.routing_sim.tt_est_std_scale,
            rng,
        )
        mdp.routing_sim.true_delivery_return[(dn, pkg)] = (td, rt)
        mdp.server.agent_prop_set[dn].at_depot = False
        mdp.server.agent_prop_set[dn].current_package = pkg
        mdp.routing_sim.busy_packages[pkg] = mdp.routing_sim.active_packages.pop(pkg)
        mdp.routing_sim.num_active_packages -= 1

    # 2) now advance one timestep in the simulator
    update_time_windows(mdp.routing_sim, mdp.server)
    update_routing_sim(mdp.routing_sim, mdp.server, rng)