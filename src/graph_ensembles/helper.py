""" This module defines a set of helper functions useful for testing.
"""

import numpy as np
import random as rd
from numba import jit


@jit(nopython=True)
def _check_unique_edges(e):
    """ Check that the edges are not repeated in the sorted edge list."""
    for i in np.arange(len(e)-1):
        if (e[i].src == e[i+1].src) and (e[i].dst == e[i+1].dst):
            assert False, 'There are repeated edges'


def _generate_id_dict(v, id_col):
    """ Return id dictionary. """
    id_dict = {}
    rep_msg = 'There is at least one repeated id in the vertices dataframe.'

    if isinstance(id_col, list):
        if len(id_col) > 1:
            # Id is a tuple
            i = 0
            for x in v[id_col].itertuples(index=False):
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        elif len(id_col) == 1:
            # Extract series
            i = 0
            for x in v[id_col[0]]:
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        else:
            # No column passed
            raise ValueError('At least one id column must be given.')

    elif isinstance(id_col, str):
        # Extract series
        i = 0
        for x in v[id_col]:
            if x in id_dict:
                raise Exception(rep_msg)
            else:
                id_dict[x] = i
                i += 1

    else:
        raise ValueError('id_col must be string or list of strings.')

    return id_dict


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


@jit(nopython=True)
def _get_strengths(edges):
    s_out = []
    s_in = []
    out_loc = {}
    in_loc = {}
    out_c = 0
    in_c = 0

    # Iterate over all non-zero edges
    for i in np.arange(len(edges)):
        # Check if index pair has already occurred
        i_out = edges['src'][i]

        if i_out not in out_loc:
            out_loc[i_out] = out_c
            out_c += 1
            s_out.append((i_out, edges['value'][i]))
        else:
            s_out[out_loc[i_out]] = (i_out, edges['value'][i] +
                                     s_out[out_loc[i_out]][1])

        # Repeat for in index
        i_in = edges['dst'][i]

        if i_in not in in_loc:
            in_loc[i_in] = in_c
            in_c += 1
            s_in.append((i_in, edges['value'][i]))
        else:
            s_in[in_loc[i_in]] = (i_in, edges['value'][i] +
                                  s_in[in_loc[i_in]][1])

    return s_out, s_in


@jit(nopython=True)
def _get_strengths_label(edges):
    s_out = []
    s_in = []
    out_loc = {}
    in_loc = {}
    out_c = 0
    in_c = 0

    # Iterate over all non-zero edges
    for i in np.arange(len(edges)):
        # Check if index pair has already occurred
        i_out = (edges['src'][i], edges['label'][i])

        if i_out not in out_loc:
            out_loc[i_out] = out_c
            out_c += 1
            s_out.append(i_out + (edges['value'][i],))
        else:
            s_out[out_loc[i_out]] = i_out + (edges['value'][i] +
                                             s_out[out_loc[i_out]][2],)

        # Repeat for in index
        i_in = (edges['dst'][i], edges['label'][i])

        if i_in not in in_loc:
            in_loc[i_in] = in_c
            in_c += 1
            s_in.append(i_in + (edges['value'][i],))
        else:
            s_in[in_loc[i_in]] = i_in + (edges['value'][i] +
                                         s_in[in_loc[i_in]][2],)

    return s_out, s_in


def get_strengths(edges, bylabel=True):
    """ Get the in and out strength sequence from the edge list provided.

    It require as input a structured array with 3 or 4 columns (src, dst,
    label, value) with the label one being optional. It returns two structured
    arrays with columns (id, label, value), with the label column being
    present if it was provided in the edge list and bylabel is True.

    Parameters
    ----------
    edges: np.ndarray
        the edge list with columns (src, dst, value) or
        (src, dst, label, value)

    bylable: bool
        flag for computing strength by label

    Returns
    -------
    out_strength: np.ndarray
        the out strength sequence

    in_strength: np.ndarray
        the in strength sequence
    """
    # # Initialise empty results
    # s_out = np.zeros((max(np.max(edges['src']), np.max(edges['dst'])),
    #                   np.max(edges['label'])))
    # s_in = np.zeros((max(np.max(edges['src']), np.max(edges['dst'])),
    #                  np.max(edges['label'])))

    if bylabel and ('label' in edges.dtype.names):
        return np.array(_get_strengths_label(edges),
                        dtype=[('id', np.int),
                               ('label', np.int),
                               ('value', np.float)])
    else:
        return np.array(_get_strengths(edges),
                        dtype=[('id', np.int),
                               ('value', np.float)])
