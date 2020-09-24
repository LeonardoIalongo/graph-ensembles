""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from scipy.optimize import fsolve


class GraphModel():
    """ General class for graph models. """

    def __init__(self, *args):
        pass


class VectorFitnessModel(GraphModel):
    """ A generalized fitness model that allows for vector strength sequences.

    Attributes
    ----------
    out_strength: np.ndarray or scipy.sparse matrix
        the out strength matrix
    in_strength: np.ndarray or scipy.sparse matrix
        the in strength matrix
    num_links: np.ndarray
        the total number of links per group
    num_nodes: np.int
        the total number of nodes
    num_groups: np.int
        the total number of groups by which the vector strengths are computed
    """

    def __init__(self, out_strength, in_strength, num_links):
        """ Return a VectorFitnessModel for the given marginal graph data.

        The assumption is that the row number of the strength matrices
        represent the node number, while the column index relates to the
        group.

        Parameters
        ----------
        out_strength: np.ndarray or scipy.sparse matrix
            the out strength matrix of a graph
        in_strength: np.ndarray or scipy.sparse matrix
            the in strength matrix of a graph
        num_links: np.ndarray
            the number of links in the graph
        param: np.ndarray
            array of parameters to be fitted by the model

        Returns
        -------
        VectorFitnessModel
            the model for the given input data
        """

        # Check that dimensions are consistent
        msg = 'In and out strength do not have the same dimensions.'
        assert in_strength.shape == out_strength.shape, msg
        msg = ('Number of groups implied by number of links input does not'
               'match strength input.')
        assert len(num_links) == out_strength.shape[1], msg

        # Initialize attributes
        self.out_strength = out_strength
        self.in_strength = in_strength
        self.num_links = num_links
        self.num_nodes = out_strength.shape[0]
        self.num_groups = out_strength.shape[1]

    def solve(self, z0=1):
        """ Fit parameters to match the ensemble to the provided data."""
        self.z = vector_density_solver(
            lambda x: vector_fitness_prob_array(
                self.out_strength,
                self.in_strength,
                x),
            self.num_links,
            z0)

    @property
    def probability_array(self):
        if hasattr(self, 'z'):
            return vector_fitness_prob_array(self.out_strength,
                                             self.in_strength,
                                             self.z)
        else:
            print('Running solver before returning matrix.')
            self.solve()
            return vector_fitness_prob_array(self.out_strength,
                                             self.in_strength,
                                             self.z)

    @property
    def probability_matrix(self):
        if hasattr(self, 'z'):
            return vector_fitness_link_prob(self.out_strength,
                                            self.in_strength,
                                            self.z)
        else:
            print('Running solver before returning matrix.')
            self.solve()
            return vector_fitness_link_prob(self.out_strength,
                                            self.in_strength,
                                            self.z)


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
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    z: np.float64
        the density parameter of the fitness model

    Returns
    -------
    numpy.ndarray
        the link probability 3d matrix

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider avoiding computation of zeros and to return
    function or iterator.
    """
    # Check that dimensions are consistent
    msg = 'In and out strength do not have the same dimensions.'
    assert in_strength.shape == out_strength.shape, msg

    # Get number of nodes and groups
    N = out_strength.shape[0]
    G = out_strength.shape[1]

    # Initialize empty result
    p = np.zeros((N, N, G), dtype=np.float64)

    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(G):
                if i != j:
                    s_i = out_strength[i, k]
                    s_j = in_strength[j, k]
                    p[i, j, k] = z*s_i*s_j / (1 + z*s_i*s_j)

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

    # Get number of nodes and groups
    N = out_strength.shape[0]
    G = out_strength.shape[1]

    # The element i,j of the matrix p here gives the probability that no link
    # of any of the G types exists between node i and j.
    # We initialize to all ones to ensure that we can multiply correctly
    p = np.ones((N, N), dtype=np.float64)

    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(G):
                if i != j:
                    s_i = out_strength[i, k]
                    s_j = in_strength[j, k]
                    p[i, j] *= 1 - (z*s_i*s_j / (1 + z*s_i*s_j))

    # Return probability of observing at least one link (out of the G types)
    return 1 - p


def vector_density_solver(p_fun, L, z0):
    """ Return the optimal z to match a given number of links vector (L).

    Parameters
    ----------
    p_fun: function
        the function returning the probability array implied by a z value
    L : np.ndarray
        number of links per group to be matched by expectation
    z0: np.float64
        initial conditions for z

    Returns
    -------
    np.float64
        the optimal z value solving L = <L>

    TODO: Currently implemented with general solver, consider iterative
    approach.
    """
    return fsolve(lambda x: np.sum(p_fun(x), axis=(0, 1)) - L, z0)
