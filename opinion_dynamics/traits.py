# opinion_dynamics/traits.py

import numpy as np
import networkx as nx

from .thesis_parameters import (
    BASE_SHARE_PROPENSITY_AUTHORITY,
    BASE_SHARE_PROPENSITY_NEUTRAL,
    BASE_VIGILANCE_AUTHORITY,
    BASE_VIGILANCE_NEUTRAL,
)


def initialize_traits_from_thesis(
    G: nx.Graph,
    authority_context: bool,
    seed: int = 42,
) -> None:
    """
    Attach thesis-based behavioural traits to each node.

    IMPORTANT:
    Route A needs these node attributes to exist for thesis NN inference:
      - vigilance_score        (1..5)
      - media_literacy         (1..5)
      - trust_authority        (1..5)
      - fam_academic_score     (1..5)

    We ALSO keep:
      - vigilance              (0..1)  for dynamics gating if you use it
      - share_propensity       (0..1)  baseline (can be overwritten by Route A)
      - is_authority           bool
    """
    rng = np.random.default_rng(seed)

    # These are your old calibrated means (already 0..1 style)
    if authority_context:
        mean_share = BASE_SHARE_PROPENSITY_AUTHORITY
        mean_vig01 = BASE_VIGILANCE_AUTHORITY
    else:
        mean_share = BASE_SHARE_PROPENSITY_NEUTRAL
        mean_vig01 = BASE_VIGILANCE_NEUTRAL

    # --- Trait distributions (placeholders but reasonable) ---
    # If your real thesis indices are 1..5-ish, this matches the training expectations.
    # Later, we can replace these with empirical sampling from your CSV distributions.
    mean_vig15 = 1.0 + 4.0 * mean_vig01  # convert 0..1 mean -> 1..5 mean

    for n in G.nodes():
        # baseline share propensity (may be overwritten by Route A thesis NN)
        G.nodes[n]["share_propensity"] = float(np.clip(rng.normal(mean_share, 0.15), 0.0, 1.0))

        # vigilance used by ABM rule (0..1)
        vig01 = float(np.clip(rng.normal(mean_vig01, 0.15), 0.0, 1.0))
        G.nodes[n]["vigilance"] = vig01

        # vigilance_score used by thesis NN (1..5)
        G.nodes[n]["vigilance_score"] = float(np.clip(rng.normal(mean_vig15, 0.9), 1.0, 5.0))

        # other thesis traits (1..5)
        G.nodes[n]["media_literacy"] = float(np.clip(rng.normal(3.2, 0.9), 1.0, 5.0))
        G.nodes[n]["trust_authority"] = float(np.clip(rng.normal(3.2, 0.9), 1.0, 5.0))
        G.nodes[n]["fam_academic_score"] = float(np.clip(rng.normal(3.0, 1.0), 1.0, 5.0))

        # default: no special authority hubs yet
        G.nodes[n]["is_authority"] = False


def set_authority_hubs(
    G: nx.Graph,
    frac_authority: float = 0.05,
    seed: int = 42,
) -> None:
    """
    Mark a fraction of nodes as 'authority' broadcasters
    and set their opinion to +1 (strongly endorsing the claim).
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if not nodes:
        return

    n_auth = max(1, int(frac_authority * len(nodes)))
    auth_nodes = rng.choice(nodes, size=n_auth, replace=False)

    for n in auth_nodes:
        G.nodes[n]["is_authority"] = True
        G.nodes[n]["opinion"] = 1.0
