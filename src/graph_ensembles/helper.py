""" This module defines a set of helper functions useful for testing.
"""

import numpy as np
import random as rd
from numba import jit


@jit(nopython=True)
def _random_edge_list(N, L, W, G=None):
    edges = []
    indexes = set()
    while len(edges) < L:
        # Generate random indices
        if G is None:
            new_index = (rd.randint(0, N-1),
                         rd.randint(0, N-1))
        else:
            new_index = (rd.randint(0, N-1),
                         rd.randint(0, N-1),
                         rd.randint(0, G-1))

        if new_index not in indexes:
            indexes.add(new_index)
            edges.append(new_index + (rd.randint(1, W),))

    return edges


def random_edge_list(N, L, W, G=None):
    """ Generate the edge list for a random graph with N nodes, L links,
    G labels, and a max link strength of W.

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

    # Initialise empty results
    s_out = np.zeros(max(np.max(edges['src']), np.max(edges['src'])))
    s_in = np.zeros(max(np.max(edges['src']), np.max(edges['src'])))

    return s_out, s_in
