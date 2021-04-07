""" This module contains any function that operates on the sGraph or
    GraphEnsemble classes or on its attributes.
"""

from . import graphs
from . import methods as mt
import numpy as np
from numba import jit
from scipy.sparse import coo_matrix
from numpy.lib.recfunctions import rec_append_fields as append_fields
import warnings


def to_sparse(coo_arr, shape, kind='coo', i_col=0, j_col=1, data_col=2):
    """ Convert to a sparse matrix the coordinate array passed.
    """
    if isinstance(i_col, int):
        i = coo_arr[coo_arr.dtype.names[i_col]]
    elif isinstance(i_col, str):
        i = coo_arr[i_col]
    elif isinstance(i_col, np.ndarray):
        i = i_col
    else:
        raise ValueError('i_col must be an int or a string.')

    if isinstance(j_col, int):
        j = coo_arr[coo_arr.dtype.names[j_col]]
    elif isinstance(j_col, str):
        j = coo_arr[j_col]
    elif isinstance(j_col, np.ndarray):
        j = j_col
    else:
        raise ValueError('j_col must be an int or a string.')

    if isinstance(data_col, int):
        data = coo_arr[coo_arr.dtype.names[data_col]]
    elif isinstance(data_col, str):
        data = coo_arr[data_col]
    elif isinstance(data_col, np.ndarray):
        data = data_col
    else:
        raise ValueError('data_col must be an int or a string.')

    mat = coo_matrix((data, (i, j)), shape=shape)

    if kind == 'coo':
        return mat
    else:
        return mat.asformat(kind)


def add_groups(g, group_dict):
    """ Add group info to a sGraph object, if already presents it raises a warning.
    """
    if hasattr(g, 'gv'):
        msg = 'Group info already present, will overwrite.'
        warnings.warn(msg, UserWarning)

    g.gv = graphs.GroupVertexList()
    g.group_dict = group_dict
    g.num_groups = len(group_dict)
    num_bytes = mt.get_num_bytes(g.num_groups)
    g.group_dtype = np.dtype('u' + str(num_bytes))
    if 'group' in g.v.dtype.names:
        g.v.group = group_dict
    else:
        g.v = append_fields(g.v, 'group', group_dict)


@jit(nopython=True)
def normalise_rows(rows, clms, weights):
    new_w = np.zeros(len(weights), dtype=weights.dtype)
    for i in range(len(rows) - 1):
        n = rows[i]
        m = rows[i + 1]
        vals = weights[n:m]

        if (n == m):
            continue

        new_w[n:m] = vals / np.sum(vals)

    return new_w


@jit(nopython=True)
def propagate_measure(indptr, indices, weights, meas, absorb=False):
    N = len(meas)
    update = np.zeros(N, dtype=np.float64)

    for i in range(len(indptr) - 1):
        n = indptr[i]
        m = indptr[i + 1]
        neighbours = indices[n:m]
        vals = weights[n:m]

        if (n == m):
            if not absorb:
                for k in range(N):
                    update[k] += meas[i]/N
        else:
            for k in range(len(neighbours)):
                j = neighbours[k]
                update[j] += vals[k]*meas[i]

    return update


def pagerank(g, alpha=0.85, max_iter=100, tol=1e-6, weighted=True):
    """ Compute the pagerank for all nodes in the graph.

    Note labelled graphs are compressed, meaning that the link multiplicity
    is eliminated by summing the weights over labels. If the graph is not
    weighted then only one link is considered to exist.
    """
    # Get adj in csr for fast propagation
    if isinstance(g, graphs.LabelGraph):
        adj = g.adjacency_matrix(kind='csr', compressed=True)
    else:
        adj = g.adjacency_matrix(kind='csr')

    if not weighted:
        adj[adj != 0] = 1
    elif weighted and not isinstance(g, graphs.WeightedGraph):
        msg = 'Weighted set to true but graph does not have weights.'
        warnings.warn(msg, UserWarning)

    # Extract data from csr matrix
    i = adj.indptr
    j = adj.indices
    w = adj.data

    # Set each row sum to be equal to one
    w = normalise_rows(i, j, w)

    # Initialise result
    N = g.num_vertices
    rank = np.ones(g.num_vertices, dtype=np.float64) / N

    # Iterate measure propagation
    for n in range(max_iter):
        # Compute update
        new_rank = propagate_measure(i, j, w, rank)

        # Add damping factor
        new_rank = new_rank*alpha + (1-alpha)/N

        # Check for convergence
        old_rank = rank
        rank = new_rank
        if np.all(np.absolute(rank - old_rank) < tol):
            print('Converged in ', n, ' iterations!')
            return rank

    print('Stopped after ', n + 1, ' iterations!')
    return rank


def trophic_depth(g, final, max_iter=100, tol=1e-3):
    """ Compute the trophic depth of each node, given the weighted graph and
    the size of the connection to the final node (which has depth zero).
    """
    # Get adj in csr for fast propagation
    if isinstance(g, graphs.LabelGraph):
        adj = g.adjacency_matrix(kind='csc', compressed=True)
    else:
        adj = g.adjacency_matrix(kind='csc')

    if not isinstance(g, graphs.WeightedGraph):
        msg = 'Graph does not have weights, will use weights of one.'
        warnings.warn(msg, UserWarning)

    # Extract data from csc matrix
    i = adj.indices
    j = adj.indptr
    w = adj.data

    # Initialise result
    strength = adj.sum(axis=1).A1 + final
    idx = strength != 0
    depth = final

    # Iterate measure propagation
    for n in range(max_iter):
        # Compute update
        new_depth = propagate_measure(j, i, w, depth + 1, absorb=True)

        # Divide by total strength
        new_depth[idx] = (final + new_depth)[idx]/strength[idx]

        # Check for convergence
        old_depth = depth
        depth = new_depth
        if np.all(np.absolute(depth - old_depth) < tol):
            print('Converged in ', n, ' iterations!')
            return depth

    print('Stopped after ', n + 1, ' iterations!')
    return depth
