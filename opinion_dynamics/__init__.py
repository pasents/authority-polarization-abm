# opinion_dynamics/__init__.py
from .network import create_social_network, rewire_homophily
from .dynamics import initialize_opinions, update_opinions
from .metrics import measure_polarization
from .simulation import SimulationParams, run_simulation
from .visualization import plot_polarization_history, plot_network

__all__ = [
    "create_social_network",
    "rewire_homophily",
    "initialize_opinions",
    "update_opinions",
    "measure_polarization",
    "SimulationParams",
    "run_simulation",
    "plot_polarization_history",
    "plot_network",
]
