""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares


class GraphModel():
    """ General class for graph models. """

    def __init__(self, *args):
        pass


class FitnessModel(GraphModel):
    pass


class StripeFitnessModel(GraphModel):
    """ A generalized fitness model that allows for vector strength sequences.

    This model allows to take into account labels of the edges and include
    this information as part of the model. The strength sequence is therefore
    now subdivided in strength per label. To quantities can be preserved by
    the ensemble: either the total number of links, or the number of links per
    label.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength matrix
    in_strength: np.ndarray
        the in strength matrix
    num_links: int (or np.ndarray)
        the total number of links (per label)
    num_nodes: np.int
        the total number of nodes
    num_labels: np.int
        the total number of labels by which the vector strengths are computed
    z: float or np.ndarray
        the vector of density parameters
    """

    def __init__(self, out_strength, in_strength, num_links):
        """ Return a VectorFitnessModel for the given marginal graph data.

        The model accepts the strength sequence as numpy arrays. The first
        column must contain the node index, the second column the label index
        to which the strength refers, and in the third column must have the
        value of the strength for the node label pair. All node label pairs
        not included are assumed zero.

        Note that the number of links given implicitly determines if the
        quantity preserved is the total number of links or the number of links
        per label. Pass only one integer for the first and a numpy array for
        the second. Note that if an array is passed then the index must be the
        same as the one in the strength sequence.

        Parameters
        ----------
        out_strength: np.ndarray
            the out strength matrix of a graph
        in_strength: np.ndarray
            the in strength matrix of a graph
        num_links: int (or np.ndarray)
            the number of links in the graph (per label)

        Returns
        -------
        VectorFitnessModel
            the model for the given input data
        """

        # Ensure that strengths passed adhere to format
        msg = 'Out strength does not have three columns.'
        assert out_strength.shape[1] == 3, msg
        msg = 'In strength does not have three columns.'
        assert in_strength.shape[1] == 3, msg

        # Get number of nodes and labels implied by the strengths
        num_nodes_out = np.max(out_strength[:, 0])
        num_nodes_in = np.max(in_strength[:, 0])
        num_nodes = max(num_nodes_out, num_nodes_in) + 1

        num_labels_out = np.max(out_strength[:, 1])
        num_labels_in = np.max(in_strength[:, 1])
        num_labels = max(num_labels_out, num_labels_in) + 1

        # Ensure that number of constraint matches number of labels
        if isinstance(num_links, np.ndarray):
            msg = ('Number of links array does not have the number of'
                   ' elements equal to the number of labels.')
            assert len(num_links) == num_labels, msg
        else:
            try:
                int(num_links)
            except TypeError:
                assert False, 'Number of links is not a number.'

        # Check that sum of in and out strengths are equal per label
        tot_out = np.zeros((num_labels))
        for row in out_strength:
            tot_out[row[1]] += row[2]
        tot_in = np.zeros((num_labels))
        for row in in_strength:
            tot_in[row[1]] += row[2]

        msg = 'Sum of strengths per label do not match.'
        assert np.all(tot_out == tot_in), msg

        # Initialize attributes
        self.out_strength = out_strength
        self.in_strength = in_strength
        self.num_links = num_links
        self.num_nodes = num_nodes
        self.num_labels = num_labels


class BlockFitnessModel(GraphModel):
    pass

    def solve(self, z0=None):
        """ Fit parameters to match the ensemble to the provided data."""
        if z0 is None:
            z0 = np.ones(self.num_groups)

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


def vector_density_solver(p_fun, L, z0):
    """ Return the optimal z to match a given number of links vector (L).

    Parameters
    ----------
    p_fun: function
        the function returning the probability array implied by a z value
    L : np.ndarray
        number of links per group to be matched by expectation
    z0: np.ndarray
        initial conditions for z vector

    Returns
    -------
    np.float64
        the optimal z value solving L = <L>

    TODO: Currently implemented with general solver, consider iterative
    approach.
    """
    return least_squares(lambda x: np.sum(p_fun(x), axis=(0, 1)) - L,
                         z0,
                         method='lm').x
