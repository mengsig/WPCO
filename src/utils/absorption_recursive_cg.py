import os
import sys
import networkx as nx

from .absorption_cg import absorption_coarse_grain


def absorption_recursive_coarse_grain(
    G, steps=1, centrality="domirank", threshold=0.5, sigma=0.99999, method="iterative"
):
    group_list = []
    for i in range(steps):
        print(f"step {i}")
        if i == 0:
            coarsened_G, groups = absorption_coarse_grain(
                G,
                centrality=centrality,
                threshold=threshold,
                sigma=sigma,
                method=method,
            )
        else:
            coarsened_G, groups = absorption_coarse_grain(
                coarsened_G,
                centrality=centrality,
                threshold=threshold,
                sigma=sigma,
                method=method,
            )

        group_list.append(groups)
        if len(coarsened_G) == 1:
            break
    groups = remapping_supernode_clusterings(group_list)
    return coarsened_G, groups


def absorption_recursive_coarse_grain_with_tree(
    G, steps=1, centrality="domirank", threshold=0.5, sigma=0.99999, method="iterative"
):
    graphs = [G]
    mappings = []

    current_G = G
    for i in range(steps):
        coarsened_G, groups = absorption_coarse_grain(
            current_G,
            centrality=centrality,
            threshold=threshold,
            sigma=sigma,
            method=method,
        )
        mappings.append(groups)
        graphs.append(coarsened_G)
        current_G = coarsened_G
        print(f"step {i}, {len(current_G)}")

    # Create hierarchical tree: leaf -> root
    tree = build_absorption_tree(mappings)

    return graphs, mappings, tree


def build_absorption_tree(mappings):
    """
    Build a hierarchical absorption tree from the list of mappings.

    Parameters
    ----------
    mappings : list of dict
        Each dict is { supernode → { node: fraction, ... } }
        mappings[0] maps fine nodes to first-level supernodes
        mappings[-1] maps previous-level supernodes to top-level supernodes

    Returns
    -------
    tree : dict
        Nested tree from top-level supernodes to original base nodes.
    """
    # Start from deepest mapping (coarsest level)
    last_mapping = mappings[-1]
    tree = {}

    for root, members in last_mapping.items():
        tree[root] = set(members.keys())  # children are supernodes from one level above

    # Walk backwards and expand
    for level in reversed(range(len(mappings) - 1)):
        current_map = mappings[level]
        new_tree = {}

        for parent, children in tree.items():
            expanded_children = set()
            for child in children:
                if child in current_map:
                    expanded_children.update(current_map[child].keys())
                else:
                    expanded_children.add(child)  # already a fine node
            new_tree[parent] = expanded_children

        tree = new_tree

    return tree


### Helper functions ####
def remapping_supernode_clusterings(supernode_clusters_array):
    """
    Expand each subsequent coarsening iteration so that by the final iteration,
    every supernode references only original nodes (with appropriate fractions).

    Parameters
    ----------
    supernode_clusters_array : list of dict
        supernode_clusters_array[i] is a dictionary:
            {
                supernode_i: {
                    node_or_supernode: fraction,
                    ...
                },
                ...
            }
        - 'supernode_i' is introduced at iteration i.
        - 'node_or_supernode' can be an original node or a supernode from iteration (i-1).
        - The sum of fractions for each supernode is 1.

    Returns
    -------
    final_dict : dict
        The last dictionary in supernode_clusters_array, but updated so all
        membership references are expanded back to the original nodes. The sum
        of fractions for each supernode remains 1.
    """

    # We expand references from iteration 0 => 1 => 2 => ... => last
    for i in range(len(supernode_clusters_array) - 1):
        prev_dict = supernode_clusters_array[i]  # iteration i
        curr_dict = supernode_clusters_array[i + 1]  # iteration i+1

        # For each supernode in iteration i+1, check its membership
        for supernode_key, membership_dict in curr_dict.items():
            # membership_dict is something like:
            #   {
            #       subnode1: fraction1,
            #       subnode2: fraction2,
            #       ...
            #   }
            # where subnodeX can be a supernode from iteration i OR an original node.
            expanded = {}  # Will become { original_node : fraction }

            for subnode, fraction_sub in membership_dict.items():
                # If 'subnode' is itself a supernode from iteration i,
                # we expand it to that supernode's original-node mapping.
                if subnode in prev_dict:
                    # subnode is a supernode from iteration i
                    # => expand subnode's dictionary of { base_node: fraction_base }
                    for base_node, fraction_base in prev_dict[subnode].items():
                        expanded[base_node] = expanded.get(base_node, 0.0) + (
                            fraction_sub * fraction_base
                        )
                else:
                    # Otherwise, subnode is an original node already; just carry the fraction
                    expanded[subnode] = expanded.get(subnode, 0.0) + fraction_sub

            # Now 'expanded' references only original nodes for this supernode_key.
            # We replace the membership dictionary with the newly expanded one:
            curr_dict[supernode_key] = expanded

    # After this loop, supernode_clusters_array[-1] only references original nodes.
    return supernode_clusters_array[-1]
