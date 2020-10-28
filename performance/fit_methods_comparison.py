import scipy.sparse as sp
import numpy as np


def vector_fitness_prob_array(out_strength, in_strength, z):
    """Compute the link probability array given the in and out strength
    sequence, and the density parameter z.

    The out and in strength sequences should be numpy arrays or scipy.sparse
    matrices of two dimension. It is assumed that the index along the
    first dimension identifies the node, while the index along the second
    dimension relates to the grouping by which the strength is computed.

    Note the returned array is a three dimensional array whose (i,j,k) element
    is the probability of observing a link of type k from node i to node j.

    Parameters
    ----------
    out_strength: scipy.sparse.csc_matrix
        the out strength sequence of graph
    in_strength: scipy.sparse.csc_matrix
        the in strength sequence of graph
    z: np.ndarray
        the group density parameter of the fitness model

    Returns
    -------
    numpy.ndarray
        the link probability 3d matrix

    """
    # Check that dimensions are consistent
    msg = 'In and out strength do not have the same dimensions.'
    assert in_strength.shape == out_strength.shape, msg

    # Check that the input is a csc_matrix
    if not isinstance(out_strength, sp.csc_matrix):
        out_strength = sp.csc_matrix(out_strength)
    if not isinstance(in_strength, sp.csc_matrix):
        in_strength = sp.csc_matrix(in_strength)

    # Get number of nodes and groups
    N = out_strength.shape[0]
    G = out_strength.shape[1]

    # Initialize empty result
    p = np.zeros((N, N, G), dtype=np.float64)

    for k in np.arange(G):
        out_index = out_strength[:, k].nonzero()[0]
        in_index = in_strength[:, k].nonzero()[0]
        out_data = out_strength[:, k].data
        in_data = in_strength[:, k].data

        for i, s_i in zip(out_index, out_data):
            for j, s_j in zip(in_index, in_data):
                if i != j:
                    p[i, j, k] = z[k]*s_i*s_j / (1 + z[k]*s_i*s_j)

    return p


def vector_fitness_link_prob(out_strength, in_strength, z):
    """ Compute the probability matrix whose (i,j) element is the probability
    of observing a link from node i to node j.

    The out and in strength sequences should be numpy arrays or scipy.sparse
    matrices of two dimension. It is assumed that the index along the
    first dimension identifies the node, while the index along the second
    dimension relates to the grouping by which the strength is computed.


    Parameters
    ----------
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    z: np.float64
        the density parameter of the fitness model

    Returns
    -------
    numpy.ndarray
        the link probability matrix

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider avoiding computation of zeros and to return
    function or iterator.

    """

    # Check that dimensions are consistent
    msg = 'In and out strength do not have the same dimensions.'
    assert in_strength.shape == out_strength.shape, msg

    # Check that the input is a csc_matrix
    if not isinstance(out_strength, sp.csc_matrix):
        out_strength = sp.csc_matrix(out_strength)
    if not isinstance(in_strength, sp.csc_matrix):
        in_strength = sp.csc_matrix(in_strength)

    # Get number of nodes and groups
    N = out_strength.shape[0]
    G = out_strength.shape[1]

    # The element i,j of the matrix p here gives the probability that no link
    # of any of the G types exists between node i and j.
    # We initialize to all ones to ensure that we can multiply correctly
    p = np.ones((N, N), dtype=np.float64)

    for k in np.arange(G):
        out_index = out_strength[:, k].nonzero()[0]
        in_index = in_strength[:, k].nonzero()[0]
        out_data = out_strength[:, k].data
        in_data = in_strength[:, k].data

        for i, s_i in zip(out_index, out_data):
            for j, s_j in zip(in_index, in_data):
                if i != j:
                    p[i, j] *= 1 - z[k]*s_i*s_j / (1 + z[k]*s_i*s_j)

    # Return probability of observing at least one link (out of the G types)
    return 1 - p
