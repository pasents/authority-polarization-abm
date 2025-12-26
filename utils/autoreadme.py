# utils/autoreadme.py
# Auto-generates a professional, research-grade README.md on every run.

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class ReadmeArtifacts:
    fig_traj: str = "figures/polarization_trajectory.png"
    fig_mech: str = "figures/mechanism_diagnostics.png"
    animation: str = "figures/authority_animation.gif"


def _to_1d_float_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[~np.isnan(arr)]


def _stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _fmt3(x: float) -> str:
    return f"{float(x):.3f}"


def write_professional_readme(
    *,
    shares_neu,
    shares_auth,
    pol_neu,
    pol_auth,
    base_params: Dict[str, Any],
    artifacts: ReadmeArtifacts = ReadmeArtifacts(),
    outpath: str = "README.md",
) -> None:
    """
    Writes a polished, researcher-grade README.md, using numbers/figures from the latest run.
    Overwrites README.md each time for a "latest run" report.
    """

    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    shares_neu = _to_1d_float_array(shares_neu)
    shares_auth = _to_1d_float_array(shares_auth)
    pol_neu = np.asarray(pol_neu, dtype=float).reshape(-1)
    pol_auth = np.asarray(pol_auth, dtype=float).reshape(-1)

    s_neu = _stats(shares_neu)
    s_auth = _stats(shares_auth)

    neu_start, neu_end = float(pol_neu[0]), float(pol_neu[-1])
    auth_start, auth_end = float(pol_auth[0]), float(pol_auth[-1])
    delta_end = auth_end - neu_end

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Keep the parameters block readable + stable
    params_json = json.dumps(base_params, indent=2)

    # Note: base_params["use_nn"] refers to *influence NN* (Route B), not the thesis share model.
    use_nn_flag = bool(base_params.get("use_nn", False))
    influence_line = "ON" if use_nn_flag else "OFF"

    readme_md = (
        "# Authority, Sharing, and Polarization  \n"
        "**An Empirically Grounded Agent-Based Model of Opinion Dynamics**\n\n"
        "---\n\n"
        "## Research Motivation\n\n"
        "Experimental evidence from my MSc thesis demonstrates that **authority framing increases individuals’ willingness "
        "to share misinformation**. While this is a micro-level behavioral effect, its **macro-level implications for "
        "polarization** are not mechanically obvious.\n\n"
        "This project evaluates the following question:\n\n"
        "> *Can authority framing causally affect polarization outcomes solely by amplifying information diffusion, "
        "even when belief susceptibility remains unchanged?*\n\n"
        "---\n\n"
        "## Conceptual Mechanism\n\n"
        "### Route A: Exposure Amplification\n\n"
        "**Authority → Increased Sharing → Higher Exposure → Altered Polarization**\n\n"
        "- Authority framing affects **sharing probability only**\n"
        "- Opinion updating rules are **identical across conditions**\n"
        "- No authority-based persuasion or influence strength is hard-coded\n\n"
        "This README reports **Route A**. A separate extension (Route B) would allow authority to modulate **susceptibility / "
        "influence strength** directly.\n\n"
        "---\n\n"
        "## Model Overview\n\n"
        "### Opinion dynamics\n"
        "- Agent-based model on a social network\n"
        "- Continuous opinions with bounded-confidence updating\n"
        "- Homophilic rewiring of network ties\n"
        "- Polarization measured as cross-sectional opinion variance\n\n"
        "### Behavioral integration (empirical)\n"
        "- Agent-level **sharing propensity** is predicted by a neural network\n"
        "- The network is trained on **real experimental data** from my MSc thesis\n"
        "- Inputs include authority context and individual traits\n"
        "- The neural network **does not** govern opinion updating\n\n"
        "### Identification strategy\n"
        "Neutral and Authority runs share:\n"
        "- the same network initialization\n"
        "- the same initial opinions\n"
        "- the same agent traits\n"
        "- the same random seed\n\n"
        "**The authority-context flag is the sole experimental manipulation.**\n\n"
        "---\n\n"
        "## Simulation Parameters (This Run)\n\n"
        "```json\n"
        f"{params_json}\n"
        "```\n\n"
        "### Model switches\n"
        f"- Thesis share model (Route A): **ON**\n"
        f"- Influence NN for opinion updating (`use_nn`, Route B): **{influence_line}**\n\n"
        "---\n\n"
        "## Learned Sharing Behavior  \n"
        "*(Neural network output)*\n\n"
        "| Condition | Mean | Std | Min | Max |\n"
        "|----------|------:|----:|----:|----:|\n"
        f"| Neutral | {_fmt3(s_neu['mean'])} | {_fmt3(s_neu['std'])} | {_fmt3(s_neu['min'])} | {_fmt3(s_neu['max'])} |\n"
        f"| Authority | {_fmt3(s_auth['mean'])} | {_fmt3(s_auth['std'])} | {_fmt3(s_auth['min'])} | {_fmt3(s_auth['max'])} |\n\n"
        "---\n\n"
        "## Polarization Outcomes\n\n"
        f"- Neutral condition (start → end): **{_fmt4(neu_start)} → {_fmt4(neu_end)}**\n"
        f"- Authority condition (start → end): **{_fmt4(auth_start)} → {_fmt4(auth_end)}**\n\n"
        f"**End-of-simulation difference (Authority − Neutral): {delta_end:+.4f}**\n\n"
        "---\n\n"
        "## Results Visualization\n\n"
        "### Polarization trajectory\n"
        f"![Polarization trajectory]({artifacts.fig_traj})\n\n"
        "### Mechanism diagnostics\n"
        f"![Mechanism diagnostics]({artifacts.fig_mech})\n\n"
        "---\n\n"
        "## Network Dynamics\n\n"
        "### Authority condition — network evolution\n"
        f"![Authority animation]({artifacts.animation})\n\n"
        "---\n\n"
        "## Interpretation (Route A)\n\n"
        "Authority framing increases sharing propensity, which increases exposure and interaction frequency. Under bounded-confidence "
        "dynamics, this can **accelerate local convergence** rather than necessarily increase fragmentation. Therefore, the sign and magnitude "
        "of polarization differences can depend on the regime (parameters, topology, and random seed).\n\n"
        "A natural next step is multi-seed replication and uncertainty quantification (e.g., confidence intervals for Δ polarization).\n\n"
        "---\n\n"
        "## Reproducibility\n\n"
        "This README was **automatically generated** from the latest simulation run.\n\n"
        f"**Timestamp:** {timestamp}\n\n"
        "---\n\n"
        "## Author\n\n"
        "**Christos**  \n"
        "MSc Behavioural Economics  \n"
        "Computational Social Science & Agent-Based Modeling\n"
    )

    out.write_text(readme_md, encoding="utf-8")
    print(f"[INFO] README.md written to {out}")
