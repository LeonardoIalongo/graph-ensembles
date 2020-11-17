""" This module defines a set of helper functions useful for testing.
"""

import numpy as np
import random as rd
from numba import jit


@jit(nopython=True)
def _random_edge_list(N, L, W, self_loops):
    edges = []
    indexes = set()

    # Check that passed value is less than max value
    if L > N*(N-1):
        L = N*(N-1)

    while len(edges) < L:
        # Generate random indices
        new_index = (rd.randint(0, N-1),
                     rd.randint(0, N-1))

        if self_loops:
            if new_index not in indexes:
                indexes.add(new_index)
                edges.append(new_index + (rd.randint(1, W),))
        else:
            if (new_index not in indexes) and (new_index[0] != new_index[1]):
                indexes.add(new_index)
                edges.append(new_index + (rd.randint(1, W),))

    return edges


@jit(nopython=True)
def _random_edge_list_labels(N, L, W, G, self_loops):
    edges = []
    indexes = set()

    # Check that passed value is less than max value
    if L > N*(N-1)*G:
        L = N*(N-1)*G

    while len(edges) < L:
        # Generate random indices
        new_index = (rd.randint(0, N-1),
                     rd.randint(0, N-1),
                     rd.randint(0, G-1))

        if self_loops:
            if new_index not in indexes:
                indexes.add(new_index)
                edges.append(new_index + (rd.randint(1, W),))
        else:
            if (new_index not in indexes) and (new_index[0] != new_index[1]):
                indexes.add(new_index)
                edges.append(new_index + (rd.randint(1, W),))

    return edges


def random_edge_list(N, L, W, G=None, self_loops=False):
    """ Generate the edge list for a random graph with N nodes, L links,
    G labels, and a max link strength of W.

    The graph can be set to allow for links from a node to itself by setting
    the self_loops flag to True.

    Parameters
    ----------
    N: int
        number of nodes
    L: int
        number of edges
    W: int
        max edge weight
    G: int
        number of edge labels

    Returns
    -------
    edges: np.ndarray
        the edge list with columns (src, dst, value) or
        (src, dst, label, value)
    """
    if G is None:
        return np.array(_random_edge_list(N, L, W, self_loops),
                        dtype=[('src', np.int),
                               ('dst', np.int),
                               ('value', np.float)])
    else:
        return np.array(_random_edge_list_labels(N, L, W, G, self_loops),
                        dtype=[('src', np.int),
                               ('dst', np.int),
                               ('label', np.int),
                               ('value', np.float)])

    # # Initialise empty results
    # s_out = np.zeros((max(np.max(edges['src']), np.max(edges['dst'])),
    #                   np.max(edges['label'])))
    # s_in = np.zeros((max(np.max(edges['src']), np.max(edges['dst'])),
    #                  np.max(edges['label'])))


# @jit(nopython=True)
def _get_strengths_label(edges):
    # Iterate over all unique values of index and label
    s_out = []
    i_out = edges[['src', 'label']].unique()

    for i in np.arange(len(i_out)):
        src = i_out[i][0]
        label = i_out[i][1]
        s_out.append((i_out[i] +
                      (np.sum(edges[(edges['src'] == src) &
                                    (edges['label'] == label)]['value']),)))

    s_in = []
    i_in = edges[['dst', 'label']].unique()

    for i in np.arange(len(i_in)):
        dst = i_in[i][0]
        label = i_in[i][1]
        s_in.append((i_in[i] +
                    (np.sum(edges[(edges['dst'] == dst) &
                                  (edges['label'] == label)]['value']),)))

    return s_out, s_in


# @jit(nopython=True)
def _get_strengths_label_dict(edges):
    s_out = []
    s_in = []
    out_loc = {}
    in_loc = {}
    out_c = 0
    in_c = 0

    # Iterate over all non-zero edges
    for i in np.arange(len(edges)):
        # Check if index pair has already occurred
        i_out = edges[['src', 'label']][i]

        if i_out not in out_loc:
            out_loc[i_out] = out_c
            out_c += 1
            s_out.append(i_out + (edges[i][3],))
        else:
            s_out[out_loc[i_out]] = i_out + (edges[i][3] +
                                             s_out[out_loc[i_out]][2],)

        # Repeat for in index
        i_in = edges[['dst', 'label']][i]

        if i_in not in in_loc:
            in_loc[i_in] = in_c
            in_c += 1
            s_in.append(i_in + (edges[i][3],))
        else:
            s_in[in_loc[i_in]][2] += edges[i][3]

        # Debug
        print(edges[i])
        print(s_out)
        print(s_in)

    return s_out, s_in


def get_strengths(edges):
    """ Get the in and out strength sequence from the edge list provided.

    It require as input a structured array with 3 or 4 columns (src, dst,
    label, value) with the label one being optional. It returns two structured
    arrays with columns (id, label, value), with the label column being
    present if it was provided in the edge list.

    Parameters
    ----------
    edges: np.ndarray
        the edge list with columns (src, dst, value) or
        (src, dst, label, value)

    Returns
    -------
    out_strength: np.ndarray
        the out strength sequence

    in_strength: np.ndarray
        the in strength sequence
    """

    if G is None:
        return np.array(_random_edge_list(N, L, W),
                        dtype=[('src', np.int),
                               ('dst', np.int),
                               ('value', np.float)])
    else:
        return np.array(_random_edge_list(N, L, W, G),
                        dtype=[('src', np.int),
                               ('dst', np.int),
                               ('label', np.int),
                               ('value', np.float)])
