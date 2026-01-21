# Authority, Sharing, and Polarization  
**An Empirically Grounded Agent-Based Model of Opinion Dynamics**

---

## Research Motivation

Experimental evidence from my MSc thesis demonstrates that **authority framing increases individuals’ willingness to share misinformation**. While this is a micro-level behavioral effect, its **macro-level implications for polarization** are not mechanically obvious.

This project evaluates the following question:

> *Can authority framing causally affect polarization outcomes solely by amplifying information diffusion, even when belief susceptibility remains unchanged?*

---

## Conceptual Mechanism

### Route A: Exposure Amplification

**Authority → Increased Sharing → Higher Exposure → Altered Polarization**

- Authority framing affects **sharing probability only**
- Opinion updating rules are **identical across conditions**
- No authority-based persuasion or influence strength is hard-coded

This README reports **Route A**. A separate extension (Route B) would allow authority to modulate **susceptibility / influence strength** directly.

---

## Model Overview

### Opinion dynamics
- Agent-based model on a social network
- Continuous opinions with bounded-confidence updating
- Homophilic rewiring of network ties
- Polarization measured as cross-sectional opinion variance

### Behavioral integration (empirical)
- Agent-level **sharing propensity** is predicted by a neural network
- The network is trained on **real experimental data** from my MSc thesis
- Inputs include authority context and individual traits
- The neural network **does not** govern opinion updating

### Identification strategy
Neutral and Authority runs share:
- the same network initialization
- the same initial opinions
- the same agent traits
- the same random seed

**The authority-context flag is the sole experimental manipulation.**

---

## Simulation Parameters (This Run)

```json
{
  "n_agents": 200,
  "avg_degree": 8,
  "n_steps": 80000,
  "polarized_start": false,
  "mu": 0.3,
  "confidence_bound": 0.6,
  "repulsion_threshold": 0.9,
  "p_rewire": 0.002,
  "similarity_threshold": 0.4,
  "seed": 42,
  "use_nn": false
}
```

### Model switches
- Thesis share model (Route A): **ON**
- Influence NN for opinion updating (`use_nn`, Route B): **OFF**

---

## Learned Sharing Behavior  
*(Neural network output)*

| Condition | Mean | Std | Min | Max |
|----------|------:|----:|----:|----:|
| Neutral | 0.306 | 0.039 | 0.205 | 0.379 |
| Authority | 0.300 | 0.036 | 0.178 | 0.370 |

---

## Polarization Outcomes

- Neutral condition (start → end): **0.3172 → 0.0481**
- Authority condition (start → end): **0.3172 → 0.0331**

**End-of-simulation difference (Authority − Neutral): -0.0150**

---

## Results Visualization

The figure below summarizes the mechanism and outcomes (including the polarization trajectory as Panel D).

![Mechanism diagnostics](figures/mechanism_diagnostics.png)

---

## Network Dynamics

### Authority condition — network evolution
![Authority animation](figures/authority_animation.gif)

---

## Interpretation (Route A)

Authority framing increases sharing propensity, which increases exposure and interaction frequency. Under bounded-confidence dynamics, this can **accelerate local convergence** rather than necessarily increase fragmentation. Therefore, the sign and magnitude of polarization differences can depend on the regime (parameters, topology, and random seed).

A natural next step is multi-seed replication and uncertainty quantification (e.g., confidence intervals for Δ polarization).

---

## Reproducibility

This README was **automatically generated** from the latest simulation run.

**Timestamp:** 2026-01-21 21:57 UTC

---

## Author

**Christos**  
MSc Behavioural Economics  
Computational Social Science & Agent-Based Modeling
