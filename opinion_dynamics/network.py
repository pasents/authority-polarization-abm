# opinion_dynamics/network.py
from typing import Dict
import numpy as np
import networkx as nx


def create_social_network(
    n_agents: int = 200,
    avg_degree: int = 6,
    seed: int = 42
) -> nx.Graph:
    """
    Create a small-world social network using the Watts–Strogatz model.
    This gives clustered structure + short path lengths (socially realistic).
    """
    np.random.seed(seed)
    # k must be even and < n_agents
    k = min(avg_degree if avg_degree % 2 == 0 else avg_degree + 1, n_agents - 1)
    G = nx.barabasi_albert_graph(n_agents, m=avg_degree // 2, seed=seed)
    return G


def rewire_homophily(
    G: nx.Graph,
    similarity_threshold: float = 0.4,
    rng: np.random.Generator | None = None
) -> None:
    """
    Echo-chamber rewiring step:
    - Pick a random node i
    - Drop a link to a neighbour whose opinion is very dissimilar
    - Create a new link to someone more similar (homophily)
    """
    if rng is None:
        rng = np.random.default_rng()

    if G.number_of_nodes() == 0:
        return

    i = rng.integers(0, G.number_of_nodes())
    if i not in G:
        return

    oi = G.nodes[i]["opinion"]
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return

    # Find dissimilar neighbours
    dissimilar_neighbors = [
        j for j in neighbors
        if abs(G.nodes[j]["opinion"] - oi) > similarity_threshold
    ]
    if not dissimilar_neighbors:
        return

    # Remove one dissimilar neighbour
    j = dissimilar_neighbors[rng.integers(0, len(dissimilar_neighbors))]
    G.remove_edge(i, j)

    # Find potential similar non-neighbours
    potential_nodes = [
        k for k in G.nodes()
        if k != i and not G.has_edge(i, k)
    ]
    if not potential_nodes:
        return

    # Sort by similarity (closest opinions first)
    potential_nodes.sort(
        key=lambda k: abs(G.nodes[k]["opinion"] - oi)
    )

    # Pick the most similar candidate (or one of the top few)
    for k in potential_nodes:
        if abs(G.nodes[k]["opinion"] - oi) <= similarity_threshold:
            G.add_edge(i, k)
            return

    # If no one is that similar, connect to the closest anyway
    k = potential_nodes[0]
    G.add_edge(i, k)
