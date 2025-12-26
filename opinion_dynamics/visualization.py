# opinion_dynamics/visualization.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np


def plot_polarization_history(pol_history):
    plt.figure()
    plt.plot(pol_history)
    plt.xlabel("Measurement step")
    plt.ylabel("Opinion variance (polarization)")
    plt.title("Polarization over time")
    plt.tight_layout()
    plt.show()


def plot_network(G: nx.Graph):
    """
    Plot the network with node color = opinion.
    """
    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)
    opinions = np.array([G.nodes[n]["opinion"] for n in G.nodes()], dtype=float)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=opinions,
        cmap=plt.cm.coolwarm,
        node_size=50,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.colorbar(nodes, label="Opinion")
    plt.axis("off")
    plt.title("Network opinions (color = opinion)")
    plt.tight_layout()
    plt.show()


def animate_network(
    G: nx.Graph,
    node_list: list,
    opinions_history: list,
    interval: int = 150,
    highlight_authority: bool = True,
    save_path: str | None = None,
    fps: int = 8,
    show: bool = True,
):
    """
    Animate the network over time.

    Parameters
    ----------
    G : networkx.Graph
        Final graph (structure used for layout).
    node_list : list
        Fixed ordering of nodes, same as used in opinions_history.
    opinions_history : list of np.ndarray
        Each array has shape (n_nodes,) with opinions at that frame.
    interval : int
        Delay between frames in milliseconds.
    highlight_authority : bool
        Draw a black ring around authority hubs.
    save_path : str | None
        If provided, saves the animation. Use ".gif" for README/GitHub.
        Example: "figures/authority_animation.gif"
    fps : int
        Frames per second for saved animation.
    show : bool
        If True, displays the animation window. Set False in automated runs.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    # Layout is fixed for all frames
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Initial frame opinions
    opinions0 = np.asarray(opinions_history[0], dtype=float)

    # Node sizes reflect share_propensity
    sizes = np.array([300 * float(G.nodes[n].get("share_propensity", 0.3)) for n in node_list], dtype=float)

    # Draw initial nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        node_color=opinions0,
        cmap=plt.cm.coolwarm,
        node_size=sizes,
        alpha=0.9,
        ax=ax,
    )
    edges = nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)

    # Highlight authority hubs
    if highlight_authority:
        authority_nodes = [n for n in node_list if G.nodes[n].get("is_authority", False)]
        if authority_nodes:
            authority_sizes = [sizes[node_list.index(n)] + 200 for n in authority_nodes]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=authority_nodes,
                node_color="none",
                edgecolors="black",
                linewidths=1.5,
                node_size=authority_sizes,
                ax=ax,
            )

    plt.colorbar(nodes, ax=ax, label="Opinion")
    ax.set_title("Network opinions over time")
    ax.axis("off")

    def update(frame_idx: int):
        opinions = np.asarray(opinions_history[frame_idx], dtype=float)
        nodes.set_array(opinions)
        ax.set_title(f"Network opinions (frame {frame_idx + 1}/{len(opinions_history)})")
        return nodes, edges

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(opinions_history),
        interval=interval,
        blit=False,
        repeat=False,
    )

    # Save if requested
    if save_path is not None:
        # Ensure folder exists
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        if save_path.lower().endswith(".gif"):
            # Pillow is the simplest for GitHub README
            try:
                from matplotlib.animation import PillowWriter
            except Exception as e:
                raise RuntimeError("PillowWriter not available. Install with: pip install pillow") from e

            ani.save(save_path, writer=PillowWriter(fps=fps))
        else:
            # Optional: support mp4 if ffmpeg is installed
            ani.save(save_path, fps=fps)

        print(f"[INFO] Saved animation to {save_path}")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani
