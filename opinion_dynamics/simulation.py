# opinion_dynamics/simulation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import copy

import numpy as np
import networkx as nx

from .network import create_social_network, rewire_homophily
from .dynamics import initialize_opinions, update_opinions
from .traits import initialize_traits_from_thesis, set_authority_hubs
from .nn_model import NNConfig, load_or_init_model


@dataclass
class SimulationParams:
    n_agents: int = 100
    avg_degree: int = 6
    n_steps: int = 50_000
    polarized_start: bool = False
    mu: float = 0.3
    confidence_bound: float = 0.6
    repulsion_threshold: float = 0.9
    p_rewire: float = 0.001
    similarity_threshold: float = 0.4
    seed: int = 42
    authority_context: bool = False

    # Optional: ABM influence NN (separate from Route A share model)
    use_nn: bool = False
    nn_model_path: str | None = None
    nn_device: str = "cpu"


# -----------------------------
# Internal helpers
# -----------------------------
def _var(G: nx.Graph) -> float:
    ops = np.array([G.nodes[n]["opinion"] for n in G.nodes()], dtype=float)
    return float(np.var(ops))


def _get_nn_model(params: SimulationParams, nn_model=None):
    if not params.use_nn:
        return None
    if nn_model is not None:
        return nn_model
    cfg = NNConfig(device=params.nn_device, model_path=params.nn_model_path)
    return load_or_init_model(cfg)


# -----------------------------
# ROUTE A: thesis NN -> share_propensity
# -----------------------------
def apply_thesis_share_propensity(
    G: nx.Graph,
    thesis_model,
    authority_context: bool,
) -> None:
    """
    Uses the thesis-trained NN to set each node's share_propensity.

    Expected node attributes (per your thesis training):
      - vigilance_score
      - media_literacy
      - trust_authority
      - fam_academic_score

    NN input: [authority_context, vigilance, media, trust, fam]
    Output: share_propensity in [0,1]
    """
    import torch

    device = next(thesis_model.parameters()).device

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return

    X = np.zeros((n, 5), dtype=np.float32)
    X[:, 0] = 1.0 if authority_context else 0.0

    # pull traits from nodes
    for k, node in enumerate(nodes):
        X[k, 1] = float(G.nodes[node].get("vigilance_score", np.nan))
        X[k, 2] = float(G.nodes[node].get("media_literacy", np.nan))
        X[k, 3] = float(G.nodes[node].get("trust_authority", np.nan))
        X[k, 4] = float(G.nodes[node].get("fam_academic_score", np.nan))

    # drop nodes with missing traits (set them to fallback later)
    mask_good = ~np.isnan(X).any(axis=1)

    # standardize cols 1..4 if model has stats
    if getattr(thesis_model, "X_mean", None) is not None and getattr(thesis_model, "X_std", None) is not None:
        X_mean = thesis_model.X_mean.detach().cpu().numpy().astype(np.float32)  # shape (1,4)
        X_std = thesis_model.X_std.detach().cpu().numpy().astype(np.float32)    # shape (1,4)
        X[mask_good, 1:] = (X[mask_good, 1:] - X_mean) / (X_std + 1e-8)

    # predict
    share = np.full((n,), 1.0, dtype=np.float32)  # fallback = always share
    if mask_good.any():
        xt = torch.from_numpy(X[mask_good]).to(device)
        with torch.no_grad():
            pred = thesis_model(xt).view(-1).detach().cpu().numpy().astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        share[mask_good] = pred

    # assign to graph
    for k, node in enumerate(nodes):
        G.nodes[node]["share_propensity"] = float(share[k])


# -----------------------------
# Graph setup (shared init helper)
# -----------------------------
def create_initialized_graph(
    params: SimulationParams,
    *,
    traits_authority_context: bool = False,
    make_authority_hubs: bool = False,
) -> nx.Graph:
    """
    Create ONE graph with opinions + traits initialized once.
    For clean comparisons, keep traits_authority_context=False.
    """
    G = create_social_network(
        n_agents=params.n_agents,
        avg_degree=params.avg_degree,
        seed=params.seed,
    )

    initialize_opinions(G, polarized=params.polarized_start, seed=params.seed)

    initialize_traits_from_thesis(
        G,
        authority_context=traits_authority_context,
        seed=params.seed,
    )

    if make_authority_hubs:
        set_authority_hubs(G, frac_authority=0.05, seed=params.seed)

    return G


# -----------------------------
# Shared init runner (Route A)
# -----------------------------
def run_simulation_with_history_from_graph(
    G_init: nx.Graph,
    params: SimulationParams,
    *,
    thesis_share_model=None,     # Route A model
    influence_nn_model=None,     # optional: separate influence NN
    record_every: int = 400,
    debug: bool = True,
):
    """
    Runs a simulation from a pre-initialized graph (shared init).

    Returns:
      - G (final graph)
      - pol_history (variance sampled over time)
      - node_list
      - opinions_history (snapshots)
      - diagnostics dict:
            checkpoints: list[int]
            applied_cum: list[int]
            avg_delta:   list[float]
            var:         list[float]
    """

    rng = np.random.default_rng(params.seed)
    G = copy.deepcopy(G_init)

    # ROUTE A: apply thesis NN to set share_propensity based on context
    if thesis_share_model is not None:
        apply_thesis_share_propensity(
            G,
            thesis_model=thesis_share_model,
            authority_context=params.authority_context,
        )

    # Optional: ABM influence NN (separate from thesis NN)
    influence_nn_model = _get_nn_model(params, nn_model=influence_nn_model)

    node_list = list(G.nodes())
    n_steps = params.n_steps
    measure_every = max(1, n_steps // 200)
    record_every = max(1, record_every)

    pol_history: List[float] = []
    opinions_history: list = []

    applied = 0
    total_delta = 0.0

    # Diagnostics (for multi-panel figure)
    diagnostics = {
        "checkpoints": [],
        "applied_cum": [],
        "avg_delta": [],
        "var": [],
    }

    start_var = _var(G)
    pol_history.append(start_var)
    opinions_history.append(np.array([G.nodes[n]["opinion"] for n in node_list], dtype=float))

    # record checkpoint at start
    diagnostics["checkpoints"].append(0)
    diagnostics["applied_cum"].append(applied)
    diagnostics["avg_delta"].append(0.0)
    diagnostics["var"].append(start_var)

    if debug:
        print(f"[DEBUG] start var={start_var:.12f}")

    for step in range(1, n_steps + 1):
        did_update, delta = update_opinions(
            G,
            mu=params.mu,
            confidence_bound=params.confidence_bound,
            repulsion_threshold=params.repulsion_threshold,
            rng=rng,
            nn_model=influence_nn_model,   # may be None (analytic influence)
        )

        if did_update:
            applied += 1
            total_delta += float(delta)

        if rng.random() < params.p_rewire:
            rewire_homophily(G, similarity_threshold=params.similarity_threshold, rng=rng)

        if step % measure_every == 0 or step == n_steps:
            pol_history.append(_var(G))

        if step % record_every == 0 or step == n_steps:
            opinions_history.append(
                np.array([G.nodes[n]["opinion"] for n in node_list], dtype=float)
            )

        # Debug checkpoints + diagnostics capture
        if debug and step % 10_000 == 0:
            cur_var = _var(G)
            avg_delta = (total_delta / applied) if applied > 0 else 0.0
            print(f"[DEBUG] step={step} var={cur_var:.12f} applied={applied} avg_delta={avg_delta:.6f}")

            diagnostics["checkpoints"].append(step)
            diagnostics["applied_cum"].append(applied)
            diagnostics["avg_delta"].append(float(avg_delta))
            diagnostics["var"].append(float(cur_var))

    end_var = _var(G)
    avg_delta = (total_delta / applied) if applied > 0 else 0.0

    if debug:
        print(f"[DEBUG] end var={end_var:.12f} applied={applied} avg_delta={avg_delta:.6f}")

    # Ensure last checkpoint exists even if n_steps not multiple of 10_000
    if len(diagnostics["checkpoints"]) == 0 or diagnostics["checkpoints"][-1] != n_steps:
        diagnostics["checkpoints"].append(n_steps)
        diagnostics["applied_cum"].append(applied)
        diagnostics["avg_delta"].append(float(avg_delta))
        diagnostics["var"].append(float(end_var))

    return G, pol_history, node_list, opinions_history, diagnostics


# -----------------------------
# Legacy functions (keep so imports don't break)
# -----------------------------
def run_simulation(params: SimulationParams, nn_model=None) -> Tuple[nx.Graph, List[float]]:
    rng = np.random.default_rng(params.seed)

    # old behavior setup
    G = create_social_network(n_agents=params.n_agents, avg_degree=params.avg_degree, seed=params.seed)
    initialize_opinions(G, polarized=params.polarized_start, seed=params.seed)
    initialize_traits_from_thesis(G, authority_context=params.authority_context, seed=params.seed)
    if params.authority_context:
        set_authority_hubs(G, frac_authority=0.05, seed=params.seed)

    nn_model = _get_nn_model(params, nn_model=nn_model)

    pol_history: List[float] = []
    measure_every = max(1, params.n_steps // 200)
    pol_history.append(_var(G))

    for step in range(1, params.n_steps + 1):
        update_opinions(
            G,
            mu=params.mu,
            confidence_bound=params.confidence_bound,
            repulsion_threshold=params.repulsion_threshold,
            rng=rng,
            nn_model=nn_model,
        )

        if rng.random() < params.p_rewire:
            rewire_homophily(G, similarity_threshold=params.similarity_threshold, rng=rng)

        if step % measure_every == 0 or step == params.n_steps:
            pol_history.append(_var(G))

    return G, pol_history


def run_simulation_with_history(params: SimulationParams, nn_model=None, record_every: int = 400):
    rng = np.random.default_rng(params.seed)

    G = create_social_network(n_agents=params.n_agents, avg_degree=params.avg_degree, seed=params.seed)
    initialize_opinions(G, polarized=params.polarized_start, seed=params.seed)
    initialize_traits_from_thesis(G, authority_context=params.authority_context, seed=params.seed)
    if params.authority_context:
        set_authority_hubs(G, frac_authority=0.05, seed=params.seed)

    nn_model = _get_nn_model(params, nn_model=nn_model)

    node_list = list(G.nodes())
    n_steps = params.n_steps
    measure_every = max(1, n_steps // 200)
    record_every = max(1, record_every)

    pol_history: List[float] = []
    opinions_history: list = []

    pol_history.append(_var(G))
    opinions_history.append(np.array([G.nodes[n]["opinion"] for n in node_list], dtype=float))

    for step in range(1, n_steps + 1):
        update_opinions(
            G,
            mu=params.mu,
            confidence_bound=params.confidence_bound,
            repulsion_threshold=params.repulsion_threshold,
            rng=rng,
            nn_model=nn_model,
        )

        if rng.random() < params.p_rewire:
            rewire_homophily(G, similarity_threshold=params.similarity_threshold, rng=rng)

        if step % measure_every == 0 or step == n_steps:
            pol_history.append(_var(G))

        if step % record_every == 0 or step == n_steps:
            opinions_history.append(
                np.array([G.nodes[n]["opinion"] for n in node_list], dtype=float)
            )

    return G, pol_history, node_list, opinions_history
