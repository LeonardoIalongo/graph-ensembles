""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from scipy.optimize import least_squares
from . import helper as hp
from . import methods
from . import iterative_models as im


class Graph():
    """ General class for graphs. """

    def __init__(self, v, e, id_col='id', src_col='src', dst_col='dst'):
        """Return a Graph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties

        Returns
        -------
        Graph
            the graph object
        """

        # Determine size of indices
        self.num_vertices = len(v.index)
        num_bytes = str(2**np.ceil(np.log2(np.log2(self.num_vertices + 1)/8)))
        self.id_dtype = np.dtype('u' + num_bytes)

        # Get dictionary of id to internal id (_id)
        self.id_dict = hp._generate_id_dict(v, id_col)

        # Generate optimized edge list
        self._e = np.rec.array(
            (e[src_col].replace(self.id_dict, inplace=False).to_numpy(),
             e[dst_col].replace(self.id_dict, inplace=False).to_numpy()),
            dtypes=[('src', self.id_dtype), ('dst', self.id_dtype)]
            )

        # Save original dataframes
        self.vertices = v
        self.edges = e


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
    num_nodes: int
        the total number of nodes
    num_labels: int
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

    def fit(self, method="least-squares", z0=None, tol=1e-8,
            eps=1e-14, max_steps=100, verbose=False):
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
            if method == "least-squares":
                self.z = least_squares(
                    lambda x: methods.expected_links_stripe_mult_z(
                        self.out_strength,
                        self.in_strength,
                        x) - self.num_links,
                    z0,
                    method='lm').x
            
            elif method == "newton":
                self.z = im.solver(z0,
                                fun = lambda x: im.loglikelihood_prime_stripe_mult_z(x,
                                                                    self.out_strength.astype(float),
                                                                    self.in_strength.astype(float),
                                                                    self.num_links),
                                fun_jac = lambda x: im.loglikelihood_hessian_stripe_mult_z(x,
                                                                    self.out_strength.astype(float),
                                                                    self.in_strength.astype(float)),
                                tol = tol,
                                eps = eps,
                                max_steps = max_steps,
                                method = "newton",
                                verbose = verbose
                )
            
            elif method == "fixed-point":
                self.z = im.solver(z0,
                                fun = lambda x: im.iterative_stripe_mult_z(x,
                                                            self.out_strength.astype(float),
                                                            self.in_strength.astype(float),
                                                            self.num_links),
                                fun_jac = None,
                                tol = tol,
                                eps = eps,
                                max_steps = max_steps,
                                method = "fixed-point",
                                verbose = verbose
                )
            
            else:
                raise ValueError("The selected method is not valid.")
        else:
            if method == "least-squares":
                self.z = least_squares(
                    lambda x: methods.expected_links_stripe_one_z(
                        self.out_strength,
                        self.in_strength,
                        x) - self.num_links,
                    z0,
                    method='lm').x
            elif method == "newton":
                self.z = im.solver(z0,
                                fun = lambda x: im.loglikelihood_prime_stripe_one_z(x,
                                                                    self.out_strength.astype(float),
                                                                    self.in_strength.astype(float),
                                                                    self.num_links),
                                fun_jac = lambda x: im.loglikelihood_hessian_stripe_one_z(x,
                                                                    self.out_strength.astype(float),
                                                                    self.in_strength.astype(float)),
                                tol = tol,
                                eps = eps,
                                max_steps = max_steps,
                                method = "newton",
                                verbose = verbose
                )
            
            elif method == "fixed-point":
                self.z = im.solver(z0,
                                fun = lambda x: im.iterative_stripe_one_z(x,
                                                            self.out_strength.astype(float),
                                                            self.in_strength.astype(float),
                                                            self.num_links),
                                fun_jac = None,
                                tol = tol,
                                eps = eps,
                                max_steps = max_steps,
                                method = "fixed-point",
                                verbose = verbose
                )

            else:
                raise ValueError("The selected method is not valid.")

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
