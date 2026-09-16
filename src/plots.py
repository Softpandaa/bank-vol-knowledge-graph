"""Single entry point. Prints every number quoted in the report and writes the figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import config
import graph
import main as m

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"]
DASH = {"crisis": (0, ()), "non-crisis": (0, (5, 2))}
COLOUR = {"crisis": OKABE[0], "non-crisis": OKABE[1]}

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def figure_embedding(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, points) in zip(axes, out["embeddings"].items()):
        labels = graph.clusters(points)
        for c in np.unique(labels):
            ax.scatter(points[labels == c, 0], points[labels == c, 1],
                       s=18, color=OKABE[c % len(OKABE)], label=f"{c + 1}")
        ax.set_xlabel("First coordinate")
        ax.set_ylabel("Second coordinate")
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=8, title="Cluster", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "embedding.png", dpi=200)
    plt.close(fig)


def figure_edge_curve(out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in out["curve"].items():
        ax.plot(list(curve.keys()), list(curve.values()), label=name.capitalize(),
                color=COLOUR[name], linestyle=DASH[name], linewidth=1.6)
    if out["crossing"] is not None:
        ax.axvline(out["crossing"], color="black", linewidth=0.8, alpha=0.6)
    for cut in (config.CUT_LOW, config.CUT_HIGH):
        ax.axvline(cut, color="black", linewidth=0.6, alpha=0.25, linestyle=":")
    ax.set_xlabel("Cut-off on the absolute partial correlation")
    ax.set_ylabel("Edges out of 300")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "edge_curve.png", dpi=200)
    plt.close(fig)


def _spring(ax, rho, points, labels, cut, title):
    """Node colour is the k means label of the PCA embedding, as in Chen and Zhang (2024)."""
    pairs, weights = graph.edges(rho, cut)
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    for (i, j), w in zip(pairs, weights):
        G.add_edge(i, j, weight=w)
    pos = nx.spring_layout(G, seed=config.RANDOM_SEED, weight="weight")
    groups = graph.clusters(points)
    nx.draw_networkx_nodes(G, pos, node_size=190, ax=ax,
                           node_color=[OKABE[g % len(OKABE)] for g in groups])
    nx.draw_networkx_labels(G, pos, labels=dict(enumerate(labels)), font_size=6.5, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           width=[G[u][v]["weight"] * 40 for u, v in G.edges()])
    ax.set_title(f"{title}, {len(pairs)} edges", fontsize=9)
    ax.axis("off")


def figure_spring(out):
    """The two subsamples at one common cut-off, so the edge counts are comparable."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, name in zip(axes, ("crisis", "non-crisis")):
        d = out["panel"][name]
        _spring(ax, d["rho"], d["pca"], config.TOP, config.CUT_LOW,
                f"{name.capitalize()}, cut-off {config.CUT_LOW:.3f}")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "spring.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = m.results()
    m.report(out)
    figure_embedding(out)
    figure_edge_curve(out)
    figure_spring(out)
    print(f"\nfigures written to {config.FIG_DIR}")
