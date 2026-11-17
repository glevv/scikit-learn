"""Building blocks of the DBCV (Density-Based Clustering Validation) index.

The quantities computed here follow the definitions of [1]_.

References
----------
.. [1] `Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
   2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
   <https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf>`_
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
from scipy.special import logsumexp


def _all_points_core_distances(distances, n_features):
    """Compute the all-points-core-distance of every point of a cluster.

    For a point `o` of a cluster of `n` points embedded in `n_features`
    dimensions, it is defined as (Definition 1 of [1]_)::

        (sum_{p != o} (1 / dist(o, p)) ** n_features / (n - 1)) ** (-1 / n_features)

    The computation is carried out in log-space: evaluating the powers directly
    overflows or underflows as soon as `n_features` gets large.

    Parameters
    ----------
    distances : ndarray of shape (cluster_size, cluster_size), dtype=float64
        Pairwise distances between the points of the cluster. The diagonal is
        ignored.

    n_features : int
        Dimension of the space the points live in.

    Returns
    -------
    core_distances : ndarray of shape (cluster_size,)
        The all-points-core-distance of each point of the cluster.
    """
    n_samples = distances.shape[0]
    with np.errstate(divide="ignore"):
        log_distances = np.log(distances)
    # A point is not its own neighbor: exp(-n_features * inf) = 0 leaves the
    # diagonal out of the sum below.
    np.fill_diagonal(log_distances, np.inf)

    if np.isneginf(log_distances).any():
        raise ValueError(
            "Identified duplicated points in a cluster. Such points cause their "
            "all-points-core-distances to be undefined, in turn affecting the whole "
            "DBCV score to be undefined for the given input. (Hint: try calling "
            "`dbcv_score` with `mst_raw_dist` set to `True`, a variant of DBCV which "
            "does not have the aforementioned limitation due to avoiding the use of "
            "core distances altogether.)"
        )

    log_distances *= -n_features
    log_mean = logsumexp(log_distances, axis=1) - np.log(n_samples - 1)
    return np.exp(-log_mean / n_features)


def _mutual_reachability(distances, core_distances_rows, core_distances_cols):
    """Compute mutual reachability distances out of distances and core distances."""
    reachability = np.maximum(distances, core_distances_rows[:, np.newaxis])
    np.maximum(reachability, core_distances_cols[np.newaxis, :], out=reachability)
    return reachability


def _max_core_to_raw_distance_ratio(distances, core_distances):
    """Largest ratio between a core distance and a raw distance of a cluster."""
    ratios = np.zeros_like(distances)
    np.divide(core_distances[np.newaxis, :], distances, out=ratios, where=distances > 0)
    return ratios.max()


def _minimum_spanning_tree(distances, tie_breaker=None):
    """Compute the minimum spanning tree of a complete graph, with Prim's algorithm.

    Contrary to the MST helpers of :mod:`sklearn.cluster`, which only need the
    weights of the edges and report them alongside the order the nodes were
    visited in, the edges returned here are the actual edges of the tree. This
    is required to tell the internal nodes from the leaves.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Weights of the edges of the complete graph. The diagonal is ignored.

    tie_breaker : ndarray of shape (n_samples, n_samples), default=None
        Secondary weights, used to choose between edges of equal weight.
        Mutual reachability distances are full of ties, since every edge
        around a point lying in a sparse area weighs the core distance of that
        point: the tree would otherwise depend on the ordering of the samples.

    Returns
    -------
    sources, targets : ndarray of shape (n_samples - 1,), dtype=intp
        The two ends of each edge of the tree.

    weights : ndarray of shape (n_samples - 1,), dtype=float64
        The weight of each edge of the tree.
    """
    n_samples = distances.shape[0]
    sources = np.zeros(n_samples - 1, dtype=np.intp)
    targets = np.empty(n_samples - 1, dtype=np.intp)
    weights = np.empty(n_samples - 1, dtype=np.float64)

    # For each node out of the tree, its closest node in the tree (starting
    # from a tree made of the first node) and how far away it is. Nodes of
    # the tree are flagged with an infinite distance.
    closest_in_tree = np.zeros(n_samples, dtype=np.intp)
    distance_to_tree = distances[0].astype(np.float64, copy=True)
    in_tree = np.zeros(n_samples, dtype=bool)
    in_tree[0] = True
    distance_to_tree[0] = np.inf
    if tie_breaker is not None:
        tie_to_tree = tie_breaker[0].astype(np.float64, copy=True)

    for i in range(n_samples - 1):
        new_node = np.argmin(distance_to_tree)
        if tie_breaker is not None:
            candidates = np.flatnonzero(distance_to_tree == distance_to_tree[new_node])
            if candidates.shape[0] > 1:
                new_node = candidates[np.argmin(tie_to_tree[candidates])]
        sources[i] = closest_in_tree[new_node]
        targets[i] = new_node
        weights[i] = distance_to_tree[new_node]

        in_tree[new_node] = True
        distance_to_tree[new_node] = np.inf
        closer = distances[new_node] < distance_to_tree
        if tie_breaker is not None:
            closer |= (distances[new_node] == distance_to_tree) & (
                tie_breaker[new_node] < tie_to_tree
            )
            closer &= ~in_tree
            tie_to_tree[closer] = tie_breaker[new_node, closer]
        else:
            closer &= ~in_tree
        distance_to_tree[closer] = distances[new_node, closer]
        closest_in_tree[closer] = new_node

    return sources, targets, weights


def _internal_minimum_spanning_tree(distances, tie_breaker=None):
    """Compute the density sparseness and the internal nodes of a cluster.

    The minimum spanning tree (MST) of the complete graph weighted by
    `distances` is built. Its internal nodes are the ones of degree greater
    than one and its internal edges are the ones joining two internal nodes.
    The density sparseness of the cluster is the largest weight of the internal
    edges (Definition 5 of [1]_).

    Parameters
    ----------
    distances : ndarray of shape (cluster_size, cluster_size), dtype=float64
        Pairwise (mutual reachability) distances between the points of the
        cluster.

    tie_breaker : ndarray of shape (cluster_size, cluster_size), default=None
        Secondary distances, used to choose between edges of equal weight when
        building the MST.

    Returns
    -------
    internal_nodes : ndarray of shape (n_internal_nodes,)
        Indices of the internal nodes of the MST. All the nodes are returned
        when the MST has no internal node, i.e. for a cluster of two points.

    density_sparseness : float
        Largest weight of the internal edges of the MST.
    """
    n_samples = distances.shape[0]
    sources, targets, weights = _minimum_spanning_tree(distances, tie_breaker)

    is_internal_node = (
        np.bincount(np.concatenate([sources, targets]), minlength=n_samples) > 1
    )
    is_internal_edge = is_internal_node[sources] & is_internal_node[targets]

    if is_internal_edge.any():
        density_sparseness = weights[is_internal_edge].max()
    else:
        # Density sparseness is not defined by the paper when the MST has no
        # internal edge. Like the MATLAB code of its authors, fall back on the
        # largest of all the edges.
        warnings.warn(
            "The minimum spanning tree has no internal edges: "
            "falling back on implementation-specific behavior "
            "by skipping edge selection.",
            UserWarning,
        )
        density_sparseness = weights.max()

    if is_internal_node.any():
        internal_nodes = np.flatnonzero(is_internal_node)
    else:
        # Keeping every node (rather than an arbitrary one) makes the result
        # independent from the ordering of the samples.
        internal_nodes = np.arange(n_samples)

    return internal_nodes, float(density_sparseness)
