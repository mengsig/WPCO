# by Marcus Engsig : @mengsig

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs as speigs
from collections import defaultdict


def absorption_coarse_grain(
    Gi, centrality="domirank", threshold=0.5, sigma=0.99999, method="iterative"
):
    """
    Coarse grain a network through a fractional absorption method.
    This preserves (as closely as possible) the total out-degree (or in-degree)
    of the original network by distributing edges among dominators in proportion
    to the product of their fractional absorptions.

    Parameters
    ----------
        Gi : nx.Graph or nx.DiGraph
            The graph to coarse grain.
        centrality : str
            The chosen centrality type to identify dominators (above cutoff).
        threshold : float
            The fraction of nodes we keep as dominators. (0 < threshold < 1)
        sigma : float or None
            For certain centralities (e.g. 'domirank'), the parameter alpha/sigma.
        method : str
            For certain centralities (e.g. 'domirank'), 'iterative' or 'analytical'.

    Returns
    -------
        coarsened_G : nx.DiGraph
            The coarsened graph (reversed before returning, so the final is reversed).
        supernode_clusters : dict
            A mapping of {dominant_node: { original_node: fraction, ... }}.
            Each dominator maps to the fractional membership of original nodes.
    """
    # 0. Checks
    G = Gi.copy()
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            "absorption_coarsening_fixed requires an nx.Graph or nx.DiGraph."
        )
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot coarsen a null graph.")
    if not (0 < threshold < 1.0):
        raise ValueError("Threshold must be in (0,1).")
    if sigma is not None and (not isinstance(sigma, float) or sigma < 0):
        raise ValueError("sigma must be None or a non-negative float.")

    # 1. Convert to directed, add weight=1 if missing
    if not nx.is_weighted(G):
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
    if not isinstance(G, nx.DiGraph):
        G = nx.to_directed(G)
    # Reverse for your conventional usage
    G = nx.reverse(G, copy=True)

    # 2. Relabel nodes to [0..N-1]
    old_nodes = list(G.nodes())
    mapping = dict(zip(old_nodes, range(len(old_nodes))))
    reverse_mapping = {v: k for k, v in mapping.items()}
    # G = nx.relabel_nodes(G, mapping)

    # 3. Compute chosen_centrality & cutoff
    if centrality == "domirank":
        # e.g. chosen_centrality, _, _ = domirank(...)
        chosen_centrality, _, _ = domirank(G, alpha=sigma, method=method)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "random":
        chosen_centrality = dict(zip(G.nodes(), np.random.rand(len(G))))
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "degree":
        adj = nx.to_scipy_sparse_array(G)
        nodes = G.nodes()
        chosen_centrality = dict(zip(nodes, adj.sum(axis=1)))
        # chosen_centrality = nx.degree_centrality(G, weight="weight")
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9

    elif centrality == "betweenness":
        chosen_centrality = nx.betweenness_centrality(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "pagerank":
        chosen_centrality = nx.pagerank(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "closeness":
        chosen_centrality = nx.closeness_centrality(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "eigenvector":
        chosen_centrality = nx.eigenvector_centrality_numpy(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "katz":
        chosen_centrality = nx.katz_centrality_numpy(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "percolation":
        chosen_centrality = nx.percolation_centrality(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "harmonic":
        chosen_centrality = nx.harmonic_centrality(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "laplacian":
        chosen_centrality = nx.laplacian_centrality(G)
        cutoff = np.percentile(list(chosen_centrality.values()), threshold * 100) + 1e-9
    elif centrality == "voterank":
        numberNodes = int((1 - threshold) * len(G))
        chosen_nodes = nx.voterank(G, number_of_nodes=numberNodes)
        chosen_centrality = {}
        for n in G.nodes():
            chosen_centrality[n] = 1.0 if n in chosen_nodes else 0.0
        cutoff = 0.5
    else:
        raise ValueError(f"Unknown centrality '{centrality}'.")

    # 4. Build the new coarsened graph
    coarsened_G = nx.DiGraph()
    supernode_clusters = defaultdict(dict)

    # 4a. Mark dominators
    dominators = [n for n, val in chosen_centrality.items() if val >= cutoff]
    for d in dominators:
        coarsened_G.add_node(d)
        supernode_clusters[d][d] = 1.0

    # 4b. Identify below-cutoff nodes
    subdominant_nodes = [n for n, val in chosen_centrality.items() if val < cutoff]

    # For each subdominant node, figure out which dominators absorb it
    absorbed = {}
    for node in subdominant_nodes:
        # Find dominators
        in_edges = G.in_edges(node, data=True)
        potential_doms = []
        for neighbor, _, wdata in in_edges:
            if chosen_centrality[neighbor] >= cutoff:
                potential_doms.append(neighbor)
        if not potential_doms:
            # No dominators => node remains alone
            coarsened_G.add_node(node)
            supernode_clusters[node][node] = 1.0
        else:
            # fractionally absorb
            sumC = sum(chosen_centrality[d] for d in potential_doms)
            fraction_map = {}
            for d in potential_doms:
                fraction_map[d] = chosen_centrality[d] / sumC
                supernode_clusters[d][node] = (
                    supernode_clusters[d].get(node, 0.0) + fraction_map[d]
                )
            absorbed[node] = (potential_doms, fraction_map)

    # 5. Add edges from each subdominant node's adjacency:
    #    including edges to dominators, edges to other subdominants, self-loops, etc.
    for node, val in chosen_centrality.items():
        if node in absorbed:
            doms, fraction_map = absorbed[node]
            # Out-neighbors
            out_edges = G.out_edges(node, data=True)
            for _, out_neighbor, wdata in out_edges:
                w = wdata["weight"]
                if out_neighbor in absorbed:
                    # out_neighbor is also subdominant => it has its own dominators + fraction_map
                    doms2, fraction_map2 = absorbed[out_neighbor]
                    # For each pair (d1, d2), add w * f1 * f2 to (d1->d2)
                    for d1 in doms:
                        f1 = fraction_map[d1]
                        for d2 in doms2:
                            f2 = fraction_map2[d2]
                            curr = (
                                coarsened_G.edges[d1, d2]["weight"]
                                if coarsened_G.has_edge(d1, d2)
                                else 0
                            )
                            coarsened_G.add_edge(d1, d2, weight=curr + w * f1 * f2)
                elif out_neighbor in dominators:
                    # out_neighbor is itself a dominator => each d in doms gets w * f_d to (d->out_neighbor)
                    for d in doms:
                        f_d = fraction_map[d]
                        curr = (
                            coarsened_G.edges[d, out_neighbor]["weight"]
                            if coarsened_G.has_edge(d, out_neighbor)
                            else 0
                        )
                        coarsened_G.add_edge(d, out_neighbor, weight=curr + w * f_d)
                else:
                    # out_neighbor is not absorbed by dominators => it's alone in coarsened_G
                    # just connect each d in doms to out_neighbor with weight = w * f_d
                    for d in doms:
                        f_d = fraction_map[d]
                        curr = (
                            coarsened_G.edges[d, out_neighbor]["weight"]
                            if coarsened_G.has_edge(d, out_neighbor)
                            else 0
                        )
                        coarsened_G.add_edge(d, out_neighbor, weight=curr + w * f_d)

        else:
            # node is a dominator or alone => we just copy edges out directly if the target is also not absorbed
            out_edges = G.out_edges(node, data=True)
            for _, out_neighbor, wdata in out_edges:
                w = wdata["weight"]
                # if out_neighbor is not in 'absorbed', just preserve
                if out_neighbor not in absorbed:
                    curr = (
                        coarsened_G.edges[node, out_neighbor]["weight"]
                        if coarsened_G.has_edge(node, out_neighbor)
                        else 0
                    )
                    coarsened_G.add_edge(node, out_neighbor, weight=curr + w)
                else:
                    # out_neighbor is subdominant => fractionally add edges node->doms2
                    doms2, fraction_map2 = absorbed[out_neighbor]
                    for d2 in doms2:
                        f2 = fraction_map2[d2]
                        curr = (
                            coarsened_G.edges[node, d2]["weight"]
                            if coarsened_G.has_edge(node, d2)
                            else 0
                        )
                        coarsened_G.add_edge(node, d2, weight=curr + w * f2)

    # 6. Reverse coarsened graph for final orientation
    coarsened_G = nx.reverse(coarsened_G, copy=True)

    # 7. Ensure every node in coarsened_G is in supernode_clusters
    for cn in coarsened_G.nodes():
        if cn not in supernode_clusters:
            supernode_clusters[cn][cn] = 1.0

    return coarsened_G, dict(supernode_clusters)


def domirank(
    G,
    method="iterative",
    alpha=0.95,
    dt=0.1,
    epsilon=1e-5,
    max_iter=1000,
    patience=10,
    max_depth=50,
):
    r"""Compute the DomiRank centrality for the graph `G`.

    DomiRank centrality [1]_ computes the centrality for a node by aggregating
    1 minus the centrality of each node in its neighborhood. This essentially finds the
    dominance of a node in its neighborhood, where the parameter $\alpha$ determines
    the amount of competition in the system by modulating $\sigma = \alpha/|\lambda_N|$.
    The competition parameter $\alpha$ tunes the balance of DomiRank centrality's
    integration of local and global topological information, to find nodes that are either
    locally or globally dominant. It is important to note that for the iterative formulation
    of DomiRank (as seen below) the competition parameter is bounded: $\sigma \in (0,1/|\lambda_N|]$.
    The DomiRank centrality of node $i$ is defined as the stationary solution to the dynamical system:

    .. math::

        d\Gamma_i(t)/dt = \sigma (d_i - \sum_j A_{ij} \Gamma_j(t)) - \Gamma_i(t),

    where $A$ is the adjacency matrix, $\lambda_N$ its smallest eigenvalue, and $d_i$ is the degree of node $i$.
    Note that the definition presented here is valid for unweighted, weighted,
    directed, and undirected networks, so in the more general case,
    a non-zero entry of the adjacency matrix $A_{ij}=w_{ij}$
    represents the existence of a link from node $i$ to node $j$
    with a weight $w_{ij}$. The steady state solution to this equation
    is computed using Newton's method. In general, one will notice that important
    nodes identified by DomiRank will be connected to a lot of other,
    unimportant nodes. However, positionally important nodes
    can also be dominated by joint-dominance of nodes, that work together
    in order to dominate the positionally important node. This centrality
    gives rise to many interesting emergent phenomena;
    see [1]_ for more information.
    DomiRank centrality can also be
    expressed in its analytical form, where the competition can now be
    supercharged - i.e. $\sigma \in [0,+\infty)$. The analytical equation
    takes the form:

    .. math::

        \boldsymbol{\Gamma} = \sigma (\sigma A + I_{N\times N})^{-1}  A  \boldsymbol{1}_{N\times 1},

    where $I$ is the identity matrix. This analytical equation can be solved
    as a linear system.

    DomiRank tends to have only positive values for relatively low
    competition levels ($\sigma \to 0$). However as the competition
    level increases, negative values might emerge. Nodes with negative
    dominance represent completely submissive nodes, which instead
    of fighting for their resources/dominance, directly give up these
    resources to their neighborhood.

    Finally, DomiRank centrality does not require the network to be weakly or strongly
    connected, and can be applied to networks with many components.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    method: string, optional (default="iterative")
        The {``"analytical"``, ``"iterative"``} method
        for computing DomiRank. Note that the computational
        time cost of the analytical method is large for
        non-regular graphs, but provides the true DomiRank.

    alpha: float, optional (default=0.95)
        The level of competition for DomiRank.

    dt: float, optional (default=0.1)
        The step size for the Newton iteration.

    epsilon: float, optional (default=1e-5)
        The relative stopping criterion for convergence.

    max_iter: integer, optional (default=100)
        Maximum number of Newton iterations allowed.
        It is recommended that ''max_iter >= 50''.

    patience: integer, optional (default=10)
        The number of steps between convergence checks.
        It is recommended that ''patience >= 10''.

    max_depth: integer, optional (default=50)
        The number of bisection steps to find the smallest
        eigenvalue. Having lower 'max_depth' reduces the
        accuracy of the smallest eigenvalue computation, but improves performance.
        It is recommended that ''max_depth > 25''.

    Returns
    -------
    nodes : dictionary
        Dictionary keyed by node with DomiRank centrality of the node as value.

    sigma : float
        $\alpha$ normalized by the smallest eigenvalue.

    converged : boolean | None
        Whether the centrality computation converged. Returns ``None`` if ``method = "analytical"``.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> centrality, sigma, converged = nx.domirank(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 -0.54', '1 1.98', '2 -1.08', '3 1.98', '4 -0.54']

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.

    NetworkXUnfeasible
        If `alpha` is negative (and thus outside its bounds): ``alpha < 0``.

        If ``alpha > 1`` when ``method = "iterative"``.

    NetworkXAlgorithmError
        If the method is not one of {``"analytical"``, ``"iterative"``}.

        If ``patience > max_iter // 3``.

        If ``max_iter < 1``.

        If 'dt' does not satisfy: ``0 < dt < 1``.

        If `epsilon` is not greater than zero: ``epsilon <= 0``.

        If ``max_depth < 1``.

    Warning
        If supercharging the competition parameter for the analytical solution: ``alpha > 1`` and ``method = "analytical"``.

        If ``method="analytical"`` for a large graph, i.e. more than ``5000`` nodes, as the algorithm will be slow.

    See Also
    --------
    :func:`~networkx.algorithms.centrality.degree_centrality`
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    References
    ----------
    .. [1] Engsig, M., Tejedor, A., Moreno, Y. et al.
        "DomiRank Centrality reveals structural fragility of complex networks via node dominance."
        Nat Commun 15, 56 (2024). https://doi.org/10.1038/s41467-023-44257-0
    """
    import numpy as np
    import scipy as sp
    import networkx as nx

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    if patience > max_iter // 3:
        raise nx.NetworkXAlgorithmError("it is mandatory that max_iter > 3 * patience")
    if max_iter < 1:
        raise nx.NetworkXAlgorithmError("it is mandatory that max_iter >= 1")
    if patience < 1:
        raise nx.NetworkXAlgorithmError("it is mandatory that patience >= 1")
    if max_depth < 1:
        raise nx.NetworkXAlgorithmError("it is mandatory that max_depth >= 1")
    if dt <= 0 or dt >= 1:
        raise nx.NetworkXAlgorithmError(
            "it is mandatory that dt be bounded such that: 0 < dt < 1"
        )
    if epsilon <= 0:
        raise nx.NetworkXAlgorithmError(
            "it is mandatory that epsilon > 0 and recommended that epsilon = 1e-5"
        )
    # G_reverse = G.reverse(copy = True)
    GAdj = nx.to_scipy_sparse_array(G)  # convert to scipy sparse csr array

    # Here we create a warning (I couldn't find a networkxwarning, only exceptions and erros), that suggests to use the iterative formulation of DomiRank rather than the analytical form.
    # Here we create another warning for alpha being supercharged
    if alpha is not None:
        if alpha > 1:
            if method == "iterative":
                raise nx.NetworkXUnfeasible(
                    "supercharging the competition parameter (alpha > 1) requires the method = 'analytical' argument"
                )
            else:
                import warnings

                warnings.warn(
                    "The competition parameter is supercharged (alpha > 1); this is only allowed because method = 'analytical'"
                )
        if alpha < 0:
            raise nx.NetworkXUnfeasible(
                "the competition parameter alpha must be positive: alpha > 0"
            )
    if GAdj.shape[0] > 5000 and method == "analytical":
        import warnings

        warnings.warn("consider using method = 'iterative' for large systems")

    # Here we renormalize alpha with the smallest eigenvalue (most negative eigenvalue) by calling the "hidden" function _find_smallest_eigenvalue()
    # Note, this function always uses the iterative definition
    if alpha is not None:
        from scipy.sparse.linalg import eigs

        if GAdj.shape[0] < 10:
            lambN, _ = eigs(GAdj.toarray(), k=1, which="SR")
        else:
            lambN, _ = eigs(GAdj, k=1, which="SR")
        sigma = np.abs(alpha / lambN)
    else:
        sigma = __optimal_sigma(
            GAdj,
            max_depth=max_depth,
            dt=0.1,
            epsilon=epsilon,
            max_iter=max_iter,
            check_step=patience,
        )[0]
    match method:
        case "analytical":
            converged = None
            psi = sp.sparse.linalg.spsolve(
                sigma * GAdj + sp.sparse.identity(GAdj.shape[0]),
                sigma * GAdj.sum(axis=-1),
            )
        case "iterative":
            psi, converged = _domirank_iterative(
                GAdj,
                sigma=sigma,
                dt=dt,
                epsilon=epsilon,
                max_iter=max_iter,
                patience=patience,
            )
        case _:
            raise nx.NetworkXUnfeasible(
                "method must be one of {'iterative', 'analytical'}"
            )
    psi = dict(zip(G, (psi).tolist()))
    return psi, sigma, converged


def _find_smallest_eigenvalue(
    G,
    min_val=0,
    max_val=1,
    max_depth=50,
    dt=0.1,
    epsilon=1e-5,
    max_iter=100,
    patience=10,
):
    """
    This function is simply used to find the smallest eigenvalue, by seeing when the DomiRank algorithm diverges. It uses
    a kind of binary search algorithm, however, with a bias to larger eigenvalues, as finding divergence is faster than
    verifying convergence.
    This function outputs the smallest eigenvalue - i.e. most negative eigenvalue.
    """
    x = (min_val + max_val) / G.sum(axis=-1).max()
    for _ in range(max_depth):
        if max_val - min_val < epsilon:
            break
        _, converged = _domirank_iterative(
            G, sigma=x, dt=dt, epsilon=epsilon, max_iter=max_iter, patience=patience
        )
        if converged:
            min_val = x
        else:
            max_val = (x + max_val) / 2
        x = (min_val + max_val) / 2
    final_val = (max_val + min_val) / 2
    return -1 / final_val


def _domirank_iterative(GAdj, sigma=0, dt=0.1, epsilon=1e-5, max_iter=100, patience=10):
    """
    This function is used for the iterative computation of DomiRank when `method = "iterative"`.
    It is also used to find the smallest eigenvalue - i.e. called in the `_find_smallest_eigenvalue()` function.
    It yields a boolean indicating convergence, and an array of the DomiRank values ordered according to `G`'s adjacency matrix.
    """
    import numpy as np

    # store this to prevent more redundant calculations in the future
    pGAdj = sigma * GAdj.astype(np.float32)
    # initialize a proportionally (to system size) small non-zero uniform vector
    psi = np.ones(pGAdj.shape[0], dtype=np.float32) / pGAdj.shape[0]
    # initialize a zero array to store values (this could be done with a smaller array with some smart indexing, but isn't computationally or memory heavy)
    max_vals = np.zeros(max_iter // patience, dtype=np.float32)
    # ensure dt is a float
    dt = np.float32(dt)
    # start a counter
    j = 0
    # set up a boundary condition for stopping divergence
    boundary = epsilon * pGAdj.shape[0] * dt
    for i in range(max_iter):
        # DomiRank iterative definition
        temp_val = ((pGAdj @ (1 - psi)) - psi) * dt
        # Newton iteration addition step
        psi += temp_val.real
        # Here we do the checking to see if we are diverging
        if i % patience == 0:
            if np.abs(temp_val).sum() < boundary:
                break
            max_vals[j] = temp_val.max()
            if j >= 2:
                if max_vals[j] > max_vals[j - 1] and max_vals[j - 1] > max_vals[j - 2]:
                    # If we are diverging, return the current step, but, with the argument that you have diverged.
                    return psi, False
            j += 1
    return psi, True


########## Here are the general functions needed for efficient dismantling and testing of networks #############


def __get_largest_component(G, strong=False):
    """
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph
    """
    import networkx as nx
    import scipy as sp

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key=len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        G = G.subgraph(GMask)
    else:
        raise TypeError("You must input a networkx.Graph Data-Type")
    return G


def __relabel_nodes(G, yield_map=False):
    """relabels the nodes to be from 0, ... len(G).
    1. Yield_map returns an extra output as a dict. in case you want to save the hash-map to retrieve node-id
    """
    import networkx as nx

    if yield_map == True:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.__relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.__relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G


def __get_component_size(G, strong=False):
    """
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph
    """
    import networkx as nx
    import scipy as sp

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key=len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        G = G.subgraph(GMask)
        return len(GMask)
    elif type(G) == sp.sparse.csr_array:
        if strong == False:
            connection_type = "weak"
        else:
            connection_type = "strong"
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(
            G, directed=True, connection=connection_type, return_labels=True
        )
        return np.bincount(lenComponent).max()
    else:
        raise TypeError(
            "You must input a networkx.Graph Data-Type or scipy.sparse.csr array"
        )


def __get_link_size(G):
    import networkx as nx
    import scipy as sp

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        links = len(G.edges())  # convert to scipy sparse if it is a graph
    elif type(G) == sp.sparse.csr_array:
        links = G.sum()
    else:
        raise TypeError("You must input a networkx.Graph Data-Type")
    return links


def __remove_node(G, removedNode):
    """
    removes the node from the graph by removing it from a networkx.Graph type, or zeroing the edges in array form.
    """
    import networkx as nx
    import scipy as sp

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        if type(removedNode) == int:
            G.__remove_node(removedNode)
        else:
            for node in removedNode:
                G.__remove_node(node)  # remove node in graph form
        return G
    elif type(G) == sp.sparse.csr_array:
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0]))
        diag[removedNode, removedNode] = (
            0  # set the rows and columns that are equal to zero in the sparse array
        )
        G = diag @ G
        return G @ diag


def __generate_attack(centrality, node_map=False):
    """we generate an attack based on a centrality measure -
    you can possibly input the node_map to convert the attack to have the correct nodeID
    """
    if node_map == False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse=True, key=zipped.get)
    return attackStrategy


def __network_attack_sampled(G, attackStrategy, sampling=0):
    """Attack a network in a sampled manner... recompute links and largest component after every xth node removal, according to some -
    G: is the input graph, preferably as a sparse array.
    inputed attack strategy
    Note: if sampling is not set, it defaults to sampling every 1%, otherwise, sampling is an integer
    that is equal to the number of nodes you want to skip every time you sample.
    So for example sampling = int(len(G)/100) would sample every 1% of the nodes removed
    """
    import networkx as nx

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G)  # convert to scipy sparse if it is a graph
    else:
        GAdj = G.copy()

    if sampling == 0:
        sampling = int(GAdj.shape[0] / 100)
    if GAdj.shape[0] < 100:
        sampling = 1
    N = GAdj.shape[0]
    initialComponent = __get_component_size(GAdj)
    initialLinks = __get_link_size(GAdj)
    m = GAdj.sum() / N
    componentEvolution = np.zeros((N // sampling + 1))
    linksEvolution = np.zeros((N // sampling) + 1)
    j = 0
    for i in range(N - 1):
        if i % sampling == 0:
            if i == 0:
                componentEvolution[j] = __get_component_size(GAdj) / initialComponent
                linksEvolution[j] = __get_link_size(GAdj) / initialLinks
                j += 1
            else:
                GAdj = __remove_node(GAdj, attackStrategy[i - sampling : i])
                componentEvolution[j] = __get_component_size(GAdj) / initialComponent
                linksEvolution[j] = __get_link_size(GAdj) / initialLinks
                j += 1
    return componentEvolution, linksEvolution


######## Beginning of __domirank stuff! ####################


def __domirank(
    G, analytical=True, sigma=-1, dt=0.1, epsilon=1e-5, max_iter=1000, check_step=10
):
    """
        G is the input graph as a (preferably) sparse array.
        This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
    complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
        Here, sigma needs to be chosen a priori.
        dt determines the step size, usually, 0.1 is sufficiently fine for most networks (could cause issues for networks
        with an extremely high degree, but has never failed me!)
        max_iter is the depth that you are searching with in case you don't converge or diverge before that.
        Checkstep is the amount of steps that you go before checking if you have converged or diverged.


        This algorithm scales with O(m) where m is the links in your sparse array.
    """
    import networkx as nx
    import scipy as sp
    import numpy as np

    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G)  # convert to scipy sparse if it is a graph
    else:
        G = G.copy()
    if analytical == False:
        if sigma == -1:
            sigma, _ = __optimal_sigma(
                G,
                analytical=False,
                dt=dt,
                epsilon=epsilon,
                max_iter=max_iter,
                check_step=check_step,
            )
        pGAdj = sigma * G.astype(np.float64)
        Psi = np.ones(pGAdj.shape[0]).astype(np.float64) / pGAdj.shape[0]
        maxVals = np.zeros(int(max_iter / check_step)).astype(np.float64)
        dt = np.float64(dt)
        j = 0
        boundary = epsilon * pGAdj.shape[0] * dt
        for i in range(max_iter):
            tempVal = ((pGAdj @ (1 - Psi)) - Psi) * dt
            Psi += tempVal.real
            if i % check_step == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                maxVals[j] = tempVal.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j - 1] and maxVals[j - 1] > maxVals[j - 2]:
                        return False, Psi
                j += 1

        return True, Psi
    else:
        if sigma == -1:
            sigma = __optimal_sigma(
                G,
                analytical=True,
                dt=dt,
                epsilon=epsilon,
                max_iter=max_iter,
                check_step=check_step,
            )
        Psi = sp.sparse.linalg.spsolve(
            sigma * G + sp.sparse.identity(G.shape[0]), sigma * G.sum(axis=-1)
        )
        return True, Psi


############## This section is for finding the optimal sigma #######################


def __process_iteration(
    q, i, analytical, sigma, sparse_array, max_iter, check_step, dt, epsilon, sampling
):
    tf, domiDist = __domirank(
        sparse_array,
        analytical=analytical,
        sigma=sigma,
        dt=dt,
        epsilon=epsilon,
        max_iter=max_iter,
        check_step=check_step,
    )
    domiAttack = __generate_attack(domiDist)
    ourTempAttack, __ = __network_attack_sampled(
        sparse_array, domiAttack, sampling=sampling
    )
    finalErrors = ourTempAttack.sum()
    q.put((i, finalErrors))


def __optimal_sigma(
    sparse_array,
    analytical=True,
    end_val=0,
    start_val=0.000001,
    iteration_no=100,
    dt=0.1,
    epsilon=1e-5,
    max_iter=100,
    check_step=10,
    max_depth=100,
    sampling=0,
):
    """
    This part finds the optimal sigma by searching the space, here are the novel parameters:
    sparse_array: is the input sparse array/matrix for the network.
    startVal: is the starting value of the space that you want to search.
    end_val: is the ending value of the space that you want to search (normally it should be the eigenvalue)
    iteration_no: the number of partitions of the space between lambN that you set

    return : the function returns the value of sigma - the numerator of the fraction of (sigma)/(-1*lambN)
    """
    import networkx as nx
    import scipy as sp

    if end_val == 0:
        from scipy.sparse.linalg import eigs

        import time

        end_val, _ = eigs(sparse_array, k=1, which="SR")
        end_val = end_val.real

    # end_val = __find_eigenvalue(
    #    sparse_array,
    #    max_depth=max_depth,
    #    dt=dt,
    #    epsilon=epsilon,
    #    max_iter=max_iter,
    #    check_step=check_step,
    # )
    import multiprocessing as mp

    endval = -1 / end_val
    tempRange = np.arange(
        start_val,
        endval + (endval - start_val) / iteration_no,
        (endval - start_val) / iteration_no,
    )
    processes = []
    q = mp.Queue()
    for i, sigma in enumerate(tempRange):
        p = mp.Process(
            target=__process_iteration,
            args=(
                q,
                i,
                analytical,
                sigma,
                sparse_array,
                max_iter,
                check_step,
                dt,
                epsilon,
                sampling,
            ),
        )
        p.start()
        processes.append(p)

    results = [None] * len(tempRange)  # Initialize a results list

    # Join the processes and gather results from the queue
    for p in processes:
        p.join()

    # Ensure that results are fetched from the queue after all processes are done
    while not q.empty():
        idx, result = q.get()
        results[idx] = result  # Store result in the correct order

    finalErrors = np.array(results)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1]
    minEig = tempRange[minEig]
    return minEig, finalErrors
