""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from scipy.optimize import least_squares
from . import methods, iterative_models


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
    now subdivided in strength per label. Two quantities can be preserved by
    the ensemble: either the total number of links, or the number of links per
    label.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength sequence
    in_strength: np.ndarray
        the in strength sequence
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
        StripeFitnessModel
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
        num_nodes = int(max(num_nodes_out, num_nodes_in) + 1)

        num_labels_out = np.max(out_strength[:, 1])
        num_labels_in = np.max(in_strength[:, 1])
        num_labels = int(max(num_labels_out, num_labels_in) + 1)

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

    def fit(self, z0=None):
        """ Compute the optimal z to match the given number of links.

        Parameters
        ----------
        z0: np.ndarray
            optional initial conditions for z vector

        TODO: Currently implemented with general solver, consider iterative
        approach.
        TODO: No checks on solver solution and convergence
        """
        if z0 is None:
            if isinstance(self.num_links, np.ndarray):
                z0 = np.ones(self.num_labels)
            else:
                z0 = 1.0

        if isinstance(self.num_links, np.ndarray):
            self.z = least_squares(
                lambda x: methods.expected_links_stripe_mult_z(
                    self.out_strength,
                    self.in_strength,
                    x) - self.num_links,
                z0,
                method='lm').x
        else:
            self.z = least_squares(
                lambda x: methods.expected_links_stripe_one_z(
                    self.out_strength,
                    self.in_strength,
                    x) - self.num_links,
                z0,
                method='lm').x

    @property
    def probability_array(self):
        """ Return the probability array of the model.

        The (i,j,k) element of this array is the probability of observing a
        link of type k from node i to j.
        """
        if not hasattr(self, 'z'):
            self.fit()

        if isinstance(self.num_links, np.ndarray):
            return methods.prob_array_stripe_mult_z(self.out_strength,
                                                    self.in_strength,
                                                    self.z,
                                                    self.num_nodes,
                                                    self.num_labels)
        else:
            return methods.prob_array_stripe_one_z(self.out_strength,
                                                   self.in_strength,
                                                   self.z,
                                                   self.num_nodes,
                                                   self.num_labels)

    @property
    def probability_matrix(self):
        """ Return the probability matrix of the model.

        The (i,j) element of this matrix is the probability of observing a
        link from node i to j of any kind.
        """
        if not hasattr(self, 'z'):
            self.fit()

        if isinstance(self.num_links, np.ndarray):
            return methods.prob_matrix_stripe_mult_z(self.out_strength,
                                                     self.in_strength,
                                                     self.z,
                                                     self.num_nodes)
        else:
            return methods.prob_matrix_stripe_one_z(self.out_strength,
                                                    self.in_strength,
                                                    self.z,
                                                    self.num_nodes)


class BlockFitnessModel(GraphModel):
    pass
