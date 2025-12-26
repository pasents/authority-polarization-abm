# opinion_dynamics/dynamics.py

from typing import Dict, Optional, Tuple
import numpy as np
import networkx as nx


def initialize_opinions(
    G: nx.Graph,
    polarized: bool = False,
    seed: int = 42
) -> Dict[int, float]:
    rng = np.random.default_rng(seed)
    opinions: Dict[int, float] = {}

    if polarized:
        nodes = list(G.nodes())
        half = len(nodes) // 2
        left = nodes[:half]
        right = nodes[half:]

        for node in left:
            opinions[node] = float(rng.normal(loc=-0.8, scale=0.1))
        for node in right:
            opinions[node] = float(rng.normal(loc=0.8, scale=0.1))
    else:
        for node in G.nodes():
            opinions[node] = float(rng.uniform(-1, 1))

    for node in opinions:
        opinions[node] = float(np.clip(opinions[node], -1, 1))

    nx.set_node_attributes(G, opinions, "opinion")
    return opinions

def update_opinions(
    G: nx.Graph,
    mu: float = 0.3,
    confidence_bound: float = 0.6,
    repulsion_threshold: float = 0.9,
    authority_boost: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    nn_model=None,
    authority_context: bool = False,   # <<< NEW
) -> Tuple[bool, float]:
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    if not nodes:
        return (False, 0.0)

    i = rng.choice(nodes)
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return (False, 0.0)

    share_i = float(G.nodes[i].get("share_propensity", 1.0))
    if rng.random() > share_i:
        return (False, 0.0)

    j = rng.choice(neighbors)

    oi = float(G.nodes[i]["opinion"])
    oj = float(G.nodes[j]["opinion"])
    diff = abs(oi - oj)
    if diff > confidence_bound:
        return (False, 0.0)

    vig_j = float(G.nodes[j].get("vigilance", 0.0))

    # IMPORTANT:
    # In the causal design, "authority_context" is the treatment.
    # Make that the signal (0/1) that the NN sees.
    auth_flag = float(authority_context)

    if nn_model is not None:
        import torch

        x = torch.tensor(
            [[oi, oj, vig_j, share_i, auth_flag]],
            dtype=torch.float32,
            device=next(nn_model.parameters()).device
        )

        with torch.no_grad():
            nn_out = float(nn_model(x).item())

        mu_eff = float(np.clip(mu * nn_out, 0.0, 1.0))

    else:
        mu_eff = mu * (1.0 - vig_j)
        if authority_context:
            mu_eff += authority_boost
        mu_eff = float(np.clip(mu_eff, 0.0, 1.0))

    if mu_eff <= 0:
        return (False, 0.0)

    new_oj = float(np.clip(oj + mu_eff * (oi - oj), -1.0, 1.0))
    delta = abs(new_oj - oj)
    G.nodes[j]["opinion"] = new_oj

    return (True, float(delta))
