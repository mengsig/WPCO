import os
import sys
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.absorption_recursive_cg import (
    absorption_recursive_coarse_grain,
    absorption_recursive_coarse_grain_with_tree,
)
from src.utils.plotting_utils import plot_defaults

plot_defaults()


@njit()
def random_seeder(dim, time_steps=10000):
    x = np.random.uniform(0, 1, (dim, dim))
    seed_pos_x = int(np.random.uniform(0, dim))
    seed_pos_y = int(np.random.uniform(0, dim))
    tele_prob = 0.001
    for i in range(time_steps):
        x[seed_pos_x, seed_pos_y] += np.random.uniform(0, 1)
        if np.random.uniform() < tele_prob:
            seed_pos_x = int(np.random.uniform(0, dim))
            seed_pos_y = int(np.random.uniform(0, dim))
        else:
            if np.random.uniform() < 0.5:
                seed_pos_x += 1
            if np.random.uniform() < 0.5:
                seed_pos_x += -1
            if np.random.uniform() < 0.5:
                seed_pos_y += 1
            if np.random.uniform() < 0.5:
                seed_pos_y += -1
            seed_pos_x = int(max(min(seed_pos_x, dim - 1), 0))
            seed_pos_y = int(max(min(seed_pos_y, dim - 1), 0))
    return x


def RemapMetrics(metric, supernode_clusters):
    """
    Compute the 'coarsened' metric for each supernode, depending on
    whether 'supernode_clusters' is in the old or new format.

    Parameters
    ----------
    metric : dict
        A dictionary mapping each original node -> some scalar value
    supernode_clusters : dict
        Old Format:  { supernode: [node1, node2, ...] }
        New Format:  { supernode: { node1: fraction1, node2: fraction2, ... } }

    Returns
    -------
    coarsened_metric : dict
        Maps supernode -> aggregated metric according to either uniform or fractional absorption.
    """
    coarsened_metric = {}

    # Special case: if no supernodes are present, return empty
    if not supernode_clusters:
        return coarsened_metric

    # Detect old vs new by looking at the first supernode's value
    sample_supernode, sample_value = next(iter(supernode_clusters.items()))

    # ---------------- OLD FORMAT (value is a list of nodes) ----------------
    if isinstance(sample_value, list):
        # First compute how many supernodes share each node
        absorbed_count = {}
        for supernode, absorbed_nodes in supernode_clusters.items():
            for node in absorbed_nodes:
                absorbed_count[node] = absorbed_count.get(node, 0) + 1

        # Initialize coarsened_metric
        for supernode, absorbed_nodes in supernode_clusters.items():
            coarsened_metric[supernode] = 0.0

        # Sum up each node's metric, divided by how many supernodes share it
        for supernode, absorbed_nodes in supernode_clusters.items():
            for node in absorbed_nodes:
                coarsened_metric[supernode] += metric[node] / absorbed_count[node]

    # ---------------- NEW FORMAT (value is dict: node -> fraction) ----------------
    elif isinstance(sample_value, dict):
        # Each supernode is a dictionary of {node: fraction, ...}
        # We multiply each fraction by that node's metric and sum.
        for supernode, membership_dict in supernode_clusters.items():
            total_value = 0.0
            for node, fraction in membership_dict.items():
                total_value += fraction * metric[node]
            coarsened_metric[supernode] = total_value

    else:
        raise ValueError(
            "Unrecognized 'supernode_clusters' format: expected either list of nodes "
            "or dict of {node: fraction} for each supernode."
        )

    return coarsened_metric


def create_adjacency(matrix):
    """
    Create a weighted graph from a 2D matrix using 4-neighbor connectivity.
    Edge weight = cell_weight * abs(cell_weight - neighbor_weight)

    Args:
        matrix (np.ndarray): 2D array representing weights.

    Returns:
        G (networkx.Graph): Undirected graph with nodes and weighted edges.
    """
    H, W = matrix.shape
    G = nx.Graph()

    percentile = np.percentile(matrix, 90)
    for i in range(H):
        for j in range(W):
            current_weight = matrix[i, j]
            if current_weight < percentile:
                continue
            current_node = (i, j)
            #            G.add_node(current_node, weight=current_weight)

            # Check and connect to 4-neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_weight = matrix[ni, nj]
                    if neighbor_weight < percentile:
                        continue
                    # TODO: what is the best weight here
                    edge_weight = (
                        current_weight**2
                        #    * 1
                        #    / (np.abs(current_weight * neighbor_weight) + 1)
                    )
                    neighbor_node = (ni, nj)
                    G.add_edge(current_node, neighbor_node, weight=edge_weight)
            # G.add_edge(current_node, current_node, weight=current_weight)

    return G


weighted_matrix = random_seeder(101, time_steps=10000)
weighted_matrix = random_seeder(101, time_steps=100000)
G = create_adjacency(weighted_matrix)
pos = {
    (i, j): (j, -i)
    for i in range(weighted_matrix.shape[0])
    for j in range(weighted_matrix.shape[1])
}

# Extract node weights for color

adj = nx.to_scipy_sparse_array(G)
node_weights = adj.sum(axis=1)
# node_weights = np.array([G.nodes[n]["weight"] for n in G.nodes()])
node_colors = node_weights  # Optional: normalize if needed

# Extract edge weights for styling
edge_weights = np.array([G.edges[e]["weight"] for e in G.edges()])
edge_colors = edge_weights
edge_widths = 1 + 2 * (edge_weights - edge_weights.min()) / (
    edge_weights.max() - edge_weights.min() + 1e-5
)

# Set up figure
fig, ax = plt.subplots(1, 3)
plt.axis("off")  # Remove axes
ax[0].set_aspect("equal")

# Draw nodes
nx.draw_networkx_nodes(
    G, pos, node_size=20, node_color=node_colors, cmap="plasma", ax=ax[0]
)

# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    alpha=0.6,
    ax=ax[0],
)

# Optional: colorbar for node weights
sm = plt.cm.ScalarMappable(
    cmap="plasma", norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max())
)
cbar = plt.colorbar(sm, ax=ax[0], shrink=0.7, pad=0.02)
cbar.set_label("Node Weight")
plot_matrix = weighted_matrix.copy()
ax[1].imshow(plot_matrix, cmap="hot")


###start of coarse grained plotting #####
graphs, mappings, tree = absorption_recursive_coarse_grain_with_tree(
    G,
    steps=6,
    threshold=0.5,
    centrality="domirank",
    sigma=0.999999999,
    method="analytical",
)
G_cg = graphs[-1]
mapping = mappings[-1]
adj = nx.to_scipy_sparse_array(G_cg)
node_weights = adj.sum(axis=1)
# node_weights = np.array([G.nodes[n]["weight"] for n in G.nodes()])
# TODO: do we normalize by length of mapping? or sum of mapping maybe?
ones = {n: 1 for n in G.nodes()}
nodes_absorbed = RemapMetrics(ones, mapping)
node_colors = np.array(
    [
        G_cg.edges[n, n]["weight"] / nodes_absorbed[n] if G_cg.has_edge(n, n) else 0.0
        for n in G_cg.nodes()
    ]
)
# Extract edge weights for styling
edge_weights = np.array([G_cg.edges[e]["weight"] for e in G_cg.edges()])
edge_colors = edge_weights
edge_widths = 1 + 2 * (edge_weights - edge_weights.min()) / (
    edge_weights.max() - edge_weights.min() + 1e-5
)

# Set up figure
plt.axis("off")  # Remove axes
ax[2].set_aspect("equal")
plt.axis("off")  # Remove axes
# Match axis limits from ax[0] to ax[2]
xlims = ax[0].get_xlim()
ylims = ax[0].get_ylim()

ax[2].set_xlim(xlims)
ax[2].set_ylim(ylims)

# Create a weighted position dictionary for G_cg based on original fine-node weights
pos_cg = {}
for supernode, members in mapping.items():
    if isinstance(members, dict):  # new format: {node: fraction}
        fine_nodes = members.items()  # (node, fraction)
    else:
        fine_nodes = [(node, 1.0) for node in members]  # treat as uniform weights

    weighted_coords = []
    total_weight = 0.0

    for node, fraction in fine_nodes:
        if node in pos:
            i, j = node
            weight = weighted_matrix[i, j] * fraction
            x, y = pos[node]
            weighted_coords.append((x * weight, y * weight))
            total_weight += weight

    if weighted_coords and total_weight > 0:
        sum_x = sum(x for x, y in weighted_coords)
        sum_y = sum(y for x, y in weighted_coords)
        pos_cg[supernode] = (sum_x / total_weight, sum_y / total_weight)

nx.draw_networkx_nodes(
    G_cg, pos_cg, node_size=20, node_color=node_colors, cmap="plasma", ax=ax[2]
)

# Draw edges
nx.draw_networkx_edges(
    G_cg,
    pos_cg,
    width=edge_widths,
    edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    alpha=0.6,
    ax=ax[2],
)

# Optional: colorbar for node weights
sm = plt.cm.ScalarMappable(
    cmap="plasma", norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max())
)
cbar = plt.colorbar(sm, ax=ax[2], shrink=0.7, pad=0.02)
cbar.set_label("Node Weight")


# Save or show
fig.tight_layout()
fig.savefig("pretty_graph.png", dpi=300)
plt.close(fig)


def plot_absorption_tree(tree, mappings, graphs, ax=None, cmap="plasma"):
    """
    Plot the absorption tree using NetworkX and Matplotlib.

    Parameters
    ----------
    tree : dict
        Dict of {supernode → set(children)} representing the hierarchy from top to bottom
    mappings : list of dict
        List of mappings from each level of coarsening, where each mapping is
        {supernode → {node: fraction, ...}}
    graphs : list of nx.Graph
        List of coarsened graphs, where graphs[i] matches mappings[i]
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, creates new figure.
    cmap : str
        Matplotlib colormap to use for node color.
    """
    # Build the tree as a directed graph
    T = nx.DiGraph()
    node_values = {}  # maps node -> selfloop / sum of fractions

    # First, collect all levels of mappings
    for level, (mapping, graph) in enumerate(
        zip(mappings, graphs[1:])
    ):  # skip original graph
        for supernode, children in mapping.items():
            sum_frac = sum(children.values())
            if graph.has_edge(supernode, supernode):
                selfloop = graph.edges[supernode, supernode]["weight"]
            else:
                selfloop = 0.0
            node_values[supernode] = selfloop / (sum_frac + 1e-8)

            # Add edge from this supernode to all absorbed items
            for child in children:
                T.add_edge(supernode, child)

    # Use Graphviz layout if available
    import networkx.drawing.nx_pydot as nx_pydot

    pos = nx_pydot.graphviz_layout(T, prog="dot")  # force DOT layout
    # Normalize node values for colormap
    node_list = list(T.nodes())
    node_colors_raw = np.array([node_values.get(n, 0.0) for n in node_list])
    node_colors_norm = (node_colors_raw - node_colors_raw.min()) / (
        node_colors_raw.max() - node_colors_raw.min() + 1e-5
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Absorption Hierarchy Tree")
    ax.axis("off")

    # Draw graph
    nx.draw(
        T,
        pos,
        ax=ax,
        with_labels=True,
        node_size=500,
        node_color=node_colors_norm,
        cmap=plt.get_cmap(cmap),
        edge_color="gray",
        font_size=8,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=node_colors_raw.min(), vmax=node_colors_raw.max()),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Self-loop / Fraction Sum", fontsize=10)


print("finished saving heatmap")
fig, ax = plt.subplots(figsize=(12, 10))
plot_absorption_tree(tree, mappings, graphs, ax=ax)
fig.tight_layout()
fig.savefig("absorption_tree.png", dpi=300)
plt.close(fig)
