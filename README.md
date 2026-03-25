## Project Overview

This is a research codebase for **decentralized multi-agent task assignment (dMTA)** — specifically, drone delivery routing with multiple depots and communication-constrained coordination. The simulation runs over San Francisco using Halton-sampled travel time estimates.

## Running the Simulation

The primary entry point is `scripts/benchmark_routing.py`. Run from the project root:

```bash
cd /path/to/dMTA
python scripts/benchmark_routing.py \
  --trials 100 --timesteps 100 \
  --n_drones 15 --n_depots 5 \
  --new_request_prob 0.5 --time_window 30 \
  --baseline ibr \
  --comms_mode 5_full \
  --init_method greedy
```

**Baseline choices:** `edd`, `hungarian`, `scoba`, `ibr`

**Communications modes** (format: `{n_depots}_{topology}`): `5_full`, `5_ring`, `5_none`, `5_edge_rm_12_T2`, etc. All topologies are defined as hard-coded dictionaries in `benchmark_routing.py:comms_graph_from_mode`.

Results are written to `results/logs/<log_dir>/` (JSON + CSVs).

**Path setup:** Scripts add `ROOT` and `ROOT/src` to `sys.path`. Imports use `domains.*`, `solver.*`, and `plotting.*` — there are no package `__init__.py` files; paths are manipulated directly.

## Architecture

### Core data flow

1. **Setup**: `setup_routing_sim()` initializes a `RoutingSimulator` (tracks packages, timings) and a `GenericAllocation` / `RoutingAllocation` (tracks drones and their assignments).
2. **Each timestep**:
   - `update_time_windows()` — rebuilds `InteractionEvent`s for each drone×package pair with estimated travel time and delivery window timestamps.
   - **Assignment algorithm** (one of the baselines below) — writes to `server.agent_task_allocation` and flips `DroneProperties.at_depot = False`.
   - `update_routing_sim()` — advances time, returns drones that have completed deliveries, generates new package requests.

### Key types (`solver/scoba_types.py`, `src/domains/routing/routing_types.py`)

- `RoutingSimulator` — world state: `active_packages`, `busy_packages`, `done_packages`, `package_claims`, `package_winners`, `true_delivery_return`
- `GenericAllocation` (`RoutingAllocation`) — agent state: `agent_set` (Drone objects), `agent_prop_set` (DroneProperties), `agent_task_windows`, `agent_task_allocation`
- `InteractionEvent` — precomputed event for one drone attempting one package: holds `travel_time` and `timestamps[MODE.START/FINISH/RETURN]`
- `SearchTree` — used by MCTS and SCoBA tree search (stored per drone in `DroneProperties.tree`)

### Assignment algorithms

| Module | Algorithm |
|--------|-----------|
| `routing_ibr.py` | **Iterative Best Response (IBR)** — main algorithm; drones iteratively pick best package given neighbors' choices, using `group_welfare` / `compute_utility`. Supports communication-restricted neighborhoods via `comms_dict`. |
| `routing_scoba.py` | **SCoBA** — constraint-based coordination with conflict resolution via `solver/scoba_conflict_resolution.py` and `solver/scoba_tree_search.py` |
| `routing_baselines.py` | **EDD** (earliest due date) and **Hungarian** (expected value Hungarian matching) |
| `mcts.py` | **MCTS** (POUCT via `pomdp_py`) — `RoutingMCTSMDP`, `RoutingAgent`, `NearestPkgPolicy` |

### Travel model (`src/domains/routing/travel_model.py`)

Global singleton initialized once via `initialize_travel_model(tree, matrix)`. Travel times use nearest-neighbor lookup in a Halton-sampled BallTree (`param_files/scoba_data.npz`). Uncertainty is modeled as Epanechnikov or Normal distribution; `delivery_success_prob_ibr()` computes P(deliver on time) for IBR, while `delivery_success_prob()` is used by SCoBA.

### Communication structure

`comms_dict` maps `depot_id -> [list of neighbor depot_ids]`. `None` = fully connected. Inside IBR, `visible_by_drone` and `visible_drones_by_depot` are derived from this. `src/domains/graph_builder.py:build_comm_structure` provides a utility wrapper.

### Package lifecycle

`active_packages` → claimed/assigned → `busy_packages` → (delivered/late) → `done_packages`. In `allow_overlap` mode, multiple drones can claim the same package (race resolved at arrival); otherwise first-come exclusive assignment.

## Parameter Files

- `param_files/sf_bb_params.toml` — bounding box (lat/lon) for 5-depot SF scenario
- `param_files/sf_bb_params_{2,6,10}dpts.toml` — variants for other depot counts (must exist for `--n_depots` to work without `--params-file`)
- `param_files/scoba_data.npz` — Halton points + travel time matrix (`points`, `estimates` arrays); invalid entries = 100000

## Logging & Output

- CSV logs are written via `scripts/csv_logger.py` (`CSVLogger`) to the run's output directory
- Key CSVs: `drone_assignment.csv`, `computational_efficiency_metrics.csv`
- Log level controlled by `--log-level`; log file: `sim_{baseline}.log` in the output directory

