# opinion_dynamics/metrics.py
from typing import Iterable
import numpy as np
import networkx as nx


def _opinions_array(G: nx.Graph) -> np.ndarray:
    """
    Helper: return node opinions as a NumPy array.
    Assumes each node has an 'opinion' attribute in [-1, 1].
    """
    return np.array([G.nodes[n]["opinion"] for n in G.nodes()], dtype=float)


def measure_polarization(G: nx.Graph) -> float:
    """
    Simple polarization measure:
    - variance of opinions across the network.
    Higher variance = more polarized.
    """
    opinions = _opinions_array(G)
    return float(np.var(opinions))


def mean_opinion(G: nx.Graph) -> float:
    """
    Average opinion in the network.
    For a misinformation claim:
      -1 = strong rejection, +1 = strong belief.
    """
    opinions = _opinions_array(G)
    return float(np.mean(opinions))


def opinion_std(G: nx.Graph) -> float:
    """
    Standard deviation of opinions.
    Similar to polarization, but in SD units.
    """
    opinions = _opinions_array(G)
    return float(np.std(opinions))


def fraction_believers(G: nx.Graph, threshold: float = 0.5) -> float:
    """
    Fraction of agents whose opinion is above a given threshold.
    For example, threshold=0.5 ≈ 'believe the claim'.
    """
    opinions = _opinions_array(G)
    return float(np.mean(opinions > threshold))
