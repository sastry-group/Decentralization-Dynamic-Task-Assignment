"""
Communication graph construction and VUG-theoretic analysis.

Builds directed communication graphs over depot/hub nodes and computes
the key quantities from Grimsman, Brown & Marden (2022) "Valid Utility
Games with Information Sharing Constraints":

  - τ(G)   : information group number       → PoA ≥ 1/(1+τ)   [Theorem 1]
  - α*(Ḡ)  : fractional independence of     → PoA ≥ 1/(1+α*)  [Theorem 2,
              the reciprocal subgraph            consistent VUGs]
  - α(G)   : independence number            → PoA ≤ 1/α       [Proposition 1]

Usage
-----
    from graph_builder import build_comms_graph, comms_dict_from_graph, graph_metrics

    G = build_comms_graph(n_depots=5, mode="ring")
    comms_dict = comms_dict_from_graph(G)          # {depot: [neighbours]}
    metrics    = graph_metrics(G)                   # dict with tau, alpha_star, poa bounds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  VUG metrics
# ---------------------------------------------------------------------------

def compute_tau(G: nx.DiGraph) -> int:
    """Number of maximal information groups τ(G).

    An information group T ⊆ V satisfies:
        ∀ i,j ∈ T :  N_i ∪ {i}  =  N_j ∪ {j}
    where N_i is the set of in-neighbours of i in G.
    τ(G) = |T(G)| where T(G) is the unique partition into maximal groups.
    """
    signatures: dict[frozenset, list] = {}
    for v in G.nodes():
        sig = frozenset(G.predecessors(v)) | {v}
        signatures.setdefault(sig, []).append(v)
    return len(signatures)


def reciprocal_subgraph(G: nx.DiGraph) -> nx.Graph:
    """Ḡ : undirected graph keeping only bidirectional edges of G."""
    G_bar = nx.Graph()
    G_bar.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if G.has_edge(v, u):
            G_bar.add_edge(u, v)
    return G_bar


def compute_alpha(G_undirected: nx.Graph) -> int:
    """Maximum independence number α(G) (exact — fine for ≤ ~20 nodes)."""
    if G_undirected.number_of_edges() == 0:
        return G_undirected.number_of_nodes()
    complement = nx.complement(G_undirected)
    clique, _ = nx.max_weight_clique(complement, weight=None)
    return len(clique)


def compute_alpha_star(G_undirected: nx.Graph) -> float:
    """Fractional independence number α*(G) via LP relaxation.

    max  1ᵀz   s.t.  Qz ≤ 1, z ≥ 0
    where Q is the clique-node incidence matrix.
    """
    nodes = sorted(G_undirected.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0

    node_idx = {v: i for i, v in enumerate(nodes)}
    cliques = list(nx.find_cliques(G_undirected))

    if not cliques:                    # no edges  →  every node is independent
        return float(n)

    Q = np.zeros((len(cliques), n))
    for ci, clique in enumerate(cliques):
        for v in clique:
            Q[ci, node_idx[v]] = 1.0

    res = linprog(
        c=-np.ones(n),
        A_ub=Q,
        b_ub=np.ones(len(cliques)),
        bounds=[(0, None)] * n,
        method="highs",
    )
    return -res.fun if res.success else float(n)


# ---------------------------------------------------------------------------
# 2.  Convenience wrappers for PoA bounds
# ---------------------------------------------------------------------------

def poa_general_vug(G: nx.DiGraph) -> float:
    """Lower bound for general VUGs:  1 / (1 + τ(G))."""
    return 1.0 / (1 + compute_tau(G))


def poa_consistent_vug(G: nx.DiGraph) -> float:
    """Lower bound for consistent VUGs (e.g. marginal-contribution):
    1 / (1 + α*(Ḡ))."""
    G_bar = reciprocal_subgraph(G)
    return 1.0 / (1 + compute_alpha_star(G_bar))


def poa_upper_bound(G: nx.DiGraph) -> float:
    """Upper bound for *any* utility design:  1 / α(Ḡ)."""
    G_bar = reciprocal_subgraph(G)
    a = compute_alpha(G_bar)
    return 1.0 / a if a > 0 else 1.0


@dataclass
class GraphMetrics:
    tau: int
    alpha_star_recip: float
    alpha_recip: int
    poa_lb_general: float
    poa_lb_consistent: float
    poa_ub: float

    def to_dict(self) -> dict:
        return {
            "tau": self.tau,
            "alpha_star_reciprocal": self.alpha_star_recip,
            "alpha_reciprocal": self.alpha_recip,
            "poa_lb_general": round(self.poa_lb_general, 6),
            "poa_lb_consistent": round(self.poa_lb_consistent, 6),
            "poa_ub": round(self.poa_ub, 6),
        }


def graph_metrics(G: nx.DiGraph) -> GraphMetrics:
    """Compute all VUG-theoretic metrics for a communication graph."""
    tau = compute_tau(G)
    G_bar = reciprocal_subgraph(G)
    a_star = compute_alpha_star(G_bar)
    a = compute_alpha(G_bar)
    return GraphMetrics(
        tau=tau,
        alpha_star_recip=a_star,
        alpha_recip=a,
        poa_lb_general=1.0 / (1 + tau),
        poa_lb_consistent=1.0 / (1 + a_star),
        poa_ub=1.0 / a if a > 0 else 1.0,
    )


# ---------------------------------------------------------------------------
# 3.  Graph generators  (any n_depots)
# ---------------------------------------------------------------------------
def build_comms_graph(n_depots: int, mode: str) -> nx.DiGraph:
    """Build a directed communication graph over depots 1..n_depots.

    Supported modes
    ---------------
    full        complete graph  (τ = 1)
    none        no edges        (τ = n)
    ring        bidirectional cycle
    chain       bidirectional path  (worst-case propagation)
    star        hub-and-spoke from depot 1
    band_<k>    each depot connects to the k nearest by index
    random_<p>  Erdős-Rényi with bidirectional edge probability p  (seed 42)
    rm_<ij>_..  remove directed edges from complete graph (e.g. rm_12_31)
    brm_<ij>_.. remove bidirectional edge pairs from complete graph (e.g. brm_12_34)
    """
    depots = list(range(1, n_depots + 1))
    G = nx.DiGraph()
    G.add_nodes_from(depots)

    if mode == "full":
        for i in depots:
            for j in depots:
                if i != j:
                    G.add_edge(i, j)

    elif mode == "none":
        pass

    elif mode == "ring":
        for i in depots:
            nxt = (i % n_depots) + 1
            prv = ((i - 2) % n_depots) + 1
            G.add_edge(i, nxt)
            G.add_edge(nxt, i)
            G.add_edge(i, prv)
            G.add_edge(prv, i)

    elif mode == "chain":
        for i in range(1, n_depots):
            G.add_edge(i, i + 1)
            G.add_edge(i + 1, i)

    elif mode == "star":
        hub = 1
        for i in depots:
            if i != hub:
                G.add_edge(hub, i)
                G.add_edge(i, hub)

    elif mode.startswith("band_"):
        k = int(mode.split("_")[1])
        for i in depots:
            for j in depots:
                if i != j and abs(i - j) <= k:
                    G.add_edge(i, j)

    elif mode.startswith("random_"):
        p = float(mode.split("_")[1])
        rng = np.random.default_rng(42)
        for i in depots:
            for j in depots:
                if i < j and rng.random() < p:
                    G.add_edge(i, j)
                    G.add_edge(j, i)

    elif mode.startswith("brm_"):
        # Bidirectional remove: start from complete, remove edge pairs
        # "brm_12_34" removes 1↔2 and 3↔4
        for i in depots:
            for j in depots:
                if i != j:
                    G.add_edge(i, j)
        for part in mode[4:].split("_"):
            u, v = int(part[0]), int(part[1])
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            if G.has_edge(v, u):
                G.remove_edge(v, u)

    elif mode.startswith("rm_"):
        # Directed remove: start from complete, remove one-way edges
        # "rm_12_31" removes 1→2 and 3→1
        for i in depots:
            for j in depots:
                if i != j:
                    G.add_edge(i, j)
        for part in mode[3:].split("_"):
            u, v = int(part[0]), int(part[1])
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    else:
        raise ValueError(
            f"Unknown comms mode '{mode}'.  Supported: "
            "full, none, ring, chain, star, band_<k>, random_<p>, "
            "rm_<ij>_.., brm_<ij>_.."
        )

    return G


# ---------------------------------------------------------------------------
# 4.  Conversion to the dict format used by the rest of the codebase
# ---------------------------------------------------------------------------

def comms_dict_from_graph(G: nx.DiGraph) -> Dict[int, List[int]]:
    """Convert a DiGraph to  {depot: sorted([neighbour_depots])}  dict.

    The convention matches the existing codebase:
      comms_dict[d] = list of depots whose actions depot d can *observe*
                    = in-neighbours of d  (predecessors in G).
    """
    return {d: sorted(G.predecessors(d)) for d in sorted(G.nodes())}


# # ---------------------------------------------------------------------------
# # 5.  Drop-in replacement for the old comms_graph_from_mode
# # ---------------------------------------------------------------------------

# def comms_graph_from_mode(mode: str, n_depots: int) -> Dict[int, List[int]]:
#     """Build a comms dict from a mode string — drop-in replacement.

#     Returns the same {depot: [neighbours]} dict the old function did,
#     but also logs the VUG metrics.
#     """
#     G = build_comms_graph(n_depots, mode)
#     metrics = graph_metrics(G)
#     logger.info(
#         "comms graph  mode=%-12s  n=%d  |  τ=%d  α*(Ḡ)=%.2f  α(Ḡ)=%d  |  "
#         "PoA(general)≥%.3f  PoA(consistent)≥%.3f  PoA(upper)≤%.3f",
#         mode, n_depots, metrics.tau, metrics.alpha_star_recip,
#         metrics.alpha_recip, metrics.poa_lb_general,
#         metrics.poa_lb_consistent, metrics.poa_ub,
#     )
#     return comms_dict_from_graph(G)


# ---------------------------------------------------------------------------
# 6.  Experiment sweep helper
# ---------------------------------------------------------------------------

def generate_experiment_configs(n_depots: int):
    """Generate a spectrum of comms graphs from dense to sparse.

    Returns a list of dicts sorted by τ, each containing:
        mode, comms_dict, and all VUG metrics.
    """
    configs = []

    # standard topologies
    for mode in ["full", "star", "ring", "chain", "none"]:
        G = build_comms_graph(n_depots, mode)
        m = graph_metrics(G)
        configs.append({
            "mode": mode,
            "comms_dict": comms_dict_from_graph(G),
            **m.to_dict(),
        })

    # band graphs with varying bandwidth
    for k in range(1, n_depots):
        mode = f"band_{k}"
        G = build_comms_graph(n_depots, mode)
        m = graph_metrics(G)
        configs.append({
            "mode": mode,
            "comms_dict": comms_dict_from_graph(G),
            **m.to_dict(),
        })

    # deduplicate (band_n-1 == full, etc.)
    seen = set()
    unique = []
    for c in configs:
        key = str(sorted(c["comms_dict"].items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return sorted(unique, key=lambda c: c["tau"])