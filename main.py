# main.py

from opinion_dynamics import SimulationParams
from opinion_dynamics.simulation import (
    create_initialized_graph,
    run_simulation_with_history_from_graph,
    apply_thesis_share_propensity,
)
from opinion_dynamics.visualization import animate_network
from opinion_dynamics.nn_model import NNConfig, load_or_init_model

import os
import copy
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_share_array(G):
    shares = np.array(
        [G.nodes[n].get("share_propensity", np.nan) for n in G.nodes()],
        dtype=float,
    )
    return shares[~np.isnan(shares)]


def summarize_share(G, label):
    shares = get_share_array(G)
    print(
        f"[DEBUG] {label} share_propensity: "
        f"mean={shares.mean():.3f} std={shares.std():.3f} "
        f"min={shares.min():.3f} max={shares.max():.3f}"
    )


def unpack_run(result):
    """
    Compatible with both return signatures:
      - (G, pol, node_list, opinions_hist)
      - (G, pol, node_list, opinions_hist, diagnostics_dict)
    """
    if len(result) == 4:
        G, pol, node_list, opinions = result
        return G, pol, node_list, opinions, None
    if len(result) == 5:
        return result
    raise ValueError("Unexpected return signature from run_simulation_with_history_from_graph")


def save_figures(
    pol_neu,
    pol_auth,
    shares_neu,
    shares_auth,
    diag_neu,
    diag_auth,
):
    os.makedirs("figures", exist_ok=True)

    # -----------------------------
    # Figure 1: Polarization trajectory (standalone)
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(pol_neu, label="Neutral", linewidth=2)
    plt.plot(pol_auth, label="Authority", linewidth=2)
    plt.xlabel("Measurement step")
    plt.ylabel("Polarization (variance)")
    plt.title("Opinion Variance Over Time (Route A: Thesis-Trained Sharing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/polarization_trajectory.png", dpi=300)
    plt.close()

    # -----------------------------
    # Figure 2: Mechanism diagnostics (multi-panel)
    # -----------------------------
    plt.figure(figsize=(12, 8))

    # Panel A: share propensity distributions (always available)
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(shares_neu, bins=30, alpha=0.7, label="Neutral")
    ax1.hist(shares_auth, bins=30, alpha=0.7, label="Authority")
    ax1.set_title("Predicted Share Propensity")
    ax1.set_xlabel("share_propensity")
    ax1.set_ylabel("count")
    ax1.legend()

    # Panels B/C: only if diagnostics exist
    has_diag = (
        isinstance(diag_neu, dict)
        and isinstance(diag_auth, dict)
        and "checkpoints" in diag_neu
        and "applied_cum" in diag_neu
        and "avg_delta" in diag_neu
        and "checkpoints" in diag_auth
        and "applied_cum" in diag_auth
        and "avg_delta" in diag_auth
    )

    ax2 = plt.subplot(2, 2, 2)
    if has_diag:
        ax2.plot(diag_neu["checkpoints"], diag_neu["applied_cum"], label="Neutral")
        ax2.plot(diag_auth["checkpoints"], diag_auth["applied_cum"], label="Authority")
        ax2.set_title("Cumulative Applied Opinion Updates")
        ax2.set_xlabel("step")
        ax2.set_ylabel("applied updates (cum)")
        ax2.legend()
    else:
        ax2.axis("off")
        ax2.text(
            0.02,
            0.5,
            "Diagnostics not returned by\nrun_simulation_with_history_from_graph.\n\n"
            "Optional upgrade:\nReturn a diagnostics dict with\n"
            "  checkpoints, applied_cum",
            fontsize=11,
            va="center",
        )

    ax3 = plt.subplot(2, 2, 3)
    if has_diag:
        ax3.plot(diag_neu["checkpoints"], diag_neu["avg_delta"], label="Neutral")
        ax3.plot(diag_auth["checkpoints"], diag_auth["avg_delta"], label="Authority")
        ax3.set_title("Average Update Magnitude")
        ax3.set_xlabel("step")
        ax3.set_ylabel("avg |Δopinion|")
        ax3.legend()
    else:
        ax3.axis("off")
        ax3.text(
            0.02,
            0.5,
            "Diagnostics not returned by\nrun_simulation_with_history_from_graph.\n\n"
            "Optional upgrade:\nReturn a diagnostics dict with\n"
            "  checkpoints, avg_delta",
            fontsize=11,
            va="center",
        )

    # Panel D: avoid duplicating Figure 1
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    ax4.text(
        0.05,
        0.55,
        "Polarization trajectory is saved as:\n\n"
        "figures/polarization_trajectory.png\n\n"
        "See Figure 1 for the standalone plot.",
        fontsize=12,
        va="center",
    )

    plt.tight_layout()
    plt.savefig("figures/mechanism_diagnostics.png", dpi=300)
    plt.close()


def main():
    set_global_seed(42)

    # Thesis-trained model (predicts share propensity from traits + authority context)
    thesis_model = load_or_init_model(
        NNConfig(device="cpu", model_path="opinion_dynamics/trained_nn_real.pt")
    )

    base_params = dict(
        n_agents=1000,
        avg_degree=8,
        n_steps=80_000,
        polarized_start=False,
        mu=0.3,
        confidence_bound=0.6,
        repulsion_threshold=0.9,
        p_rewire=0.002,
        similarity_threshold=0.4,
        seed=42,
        use_nn=False,  # influence NN OFF (Route B not used)
    )

    neutral_params = SimulationParams(**base_params, authority_context=False)
    authority_params = SimulationParams(**base_params, authority_context=True)

    # Shared init: same network, same opinions, same traits
    G_init = create_initialized_graph(
        neutral_params,
        traits_authority_context=False,
        make_authority_hubs=False,
    )

    # Debug: verify thesis NN changes share_propensity by authority flag
    Gn = copy.deepcopy(G_init)
    Ga = copy.deepcopy(G_init)
    apply_thesis_share_propensity(Gn, thesis_model, False)
    apply_thesis_share_propensity(Ga, thesis_model, True)
    summarize_share(Gn, "Neutral")
    summarize_share(Ga, "Authority")

    shares_neu = get_share_array(Gn)
    shares_auth = get_share_array(Ga)

    # Run simulations (shared init)
    res_neu = run_simulation_with_history_from_graph(
        G_init,
        neutral_params,
        thesis_share_model=thesis_model,
        debug=True,
    )
    G_neu, pol_neu, nodes, ops_neu, diag_neu = unpack_run(res_neu)

    res_auth = run_simulation_with_history_from_graph(
        G_init,
        authority_params,
        thesis_share_model=thesis_model,
        debug=True,
    )
    G_auth, pol_auth, _, ops_auth, diag_auth = unpack_run(res_auth)

    # Save figures for README
    save_figures(
        pol_neu,
        pol_auth,
        shares_neu,
        shares_auth,
        diag_neu,
        diag_auth,
    )

    # Save animation GIF for README (requires animate_network(save_path=..., fps=..., show=...))
    animate_network(
        G_auth,
        nodes,
        ops_auth,
        interval=150,
        highlight_authority=True,
        save_path="figures/authority_animation.gif",
        fps=8,
        show=False,
    )

    # Professional auto README
    from utils.autoreadme import write_professional_readme, ReadmeArtifacts

    write_professional_readme(
        shares_neu=shares_neu,
        shares_auth=shares_auth,
        pol_neu=pol_neu,
        pol_auth=pol_auth,
        base_params=base_params,
        artifacts=ReadmeArtifacts(
            fig_traj="figures/polarization_trajectory.png",
            fig_mech="figures/mechanism_diagnostics.png",
            animation="figures/authority_animation.gif",
        ),
        outpath="README.md",
    )

    print("[INFO] Done. Figures saved in ./figures, GIF saved, and README.md updated.")


if __name__ == "__main__":
    main()
