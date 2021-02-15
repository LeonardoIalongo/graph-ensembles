""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from numpy.lib.recfunctions import rec_append_fields as append_fields
import pandas as pd
from scipy.optimize import least_squares
from . import methods as mt
from . import iterative_models as im
import warnings


class Graph():
    """ Generator class for graphs. Returns correct object depending on inputs
    passed.
    """
    def __new__(cls, v, e, **kwargs):
        # Ensure passed arguments are accepted
        allowed_arguments = ['v_id', 'src', 'dst', 'weight', 'edge_label']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)

        if 'v_id' not in kwargs:
            id_col = 'id'
        else:
            id_col = kwargs['v_id']

        if 'src' not in kwargs:
            src_col = 'src'
        else:
            src_col = kwargs['src']

        if 'dst' not in kwargs:
            dst_col = 'dst'
        else:
            dst_col = kwargs['dst']

        if 'weight' in kwargs:
            weight_col = kwargs['weight']
            if 'edge_label' in kwargs:
                label_col = kwargs['edge_label']
                return WeightedEdgelabelGraph(v,
                                              e,
                                              v_id=id_col,
                                              src=src_col,
                                              dst=dst_col,
                                              weight=weight_col,
                                              edge_label=label_col)
            else:
                return WeightedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col,
                                     weight=weight_col)

        else:
            if 'edge_label' in kwargs:
                label_col = kwargs['edge_label']
                return EdgelabelGraph(v,
                                      e,
                                      v_id=id_col,
                                      src=src_col,
                                      dst=dst_col,
                                      edge_label=label_col)
            else:
                return DirectedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col)


class sGraph():
    """ General class for graphs.
    """
    pass


class DirectedGraph(sGraph):
    """ General class for directed graphs.

    Attributes
    ----------
    num_vertices: int
        number of vertices in the graph
    num_edges: int
        number of edges in the graph
    v: numpy.rec.array
        array containing the computed properties of the vertices
    e: numpy.rec.array
        array containing the edge list in a condensed format
    id_dict: dict
        dictionary to convert original identifiers to positions in v
    id_type: numpy.dtype
        type of the id (e.g. np.uint16)
    sort_ind: numpy.array
        the index used for sorting e

    Methods
    -------
    degree:
        compute the undirected degree sequence
    out_degree:
        compute the out degree sequence
    in_degree:
        compute the in degree sequence
    """

    def __init__(self, v, e, v_id, src, dst):
        """Return a sGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str or list of str
            identifier column for the source vertex
        dst: str or list of str
            identifier column for the destination vertex

        Returns
        -------
        sGraph
            the graph object
        """

        assert isinstance(v, pd.DataFrame), 'Only dataframe input supported.'
        assert isinstance(e, pd.DataFrame), 'Only dataframe input supported.'

        # Determine size of indices
        self.num_vertices = len(v.index)
        self.num_edges = len(e.index)
        num_bytes = mt._get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype('u' + str(num_bytes))

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        self.id_dict = mt._generate_id_dict(v, v_id)

        # Check that no vertex id in e is not present in v
        # and generate optimized edge list
        smsg = 'Some source vertices are not in v.'
        dmsg = 'Some destination vertices are not in v.'
        if isinstance(src, list) and isinstance(dst, list):
            n = len(src)
            m = len(dst)
            src_array = np.empty(self.num_edges, dtype=self.id_dtype)
            dst_array = np.empty(self.num_edges, dtype=self.id_dtype)
            i = 0
            for row in e[src + dst].itertuples(index=False):
                row_src = row[0:n]
                row_dst = row[n:n+m]
                if row_src not in self.id_dict:
                    assert False, smsg
                if row_dst not in self.id_dict:
                    assert False, dmsg
                src_array[i] = self.id_dict[row_src]
                dst_array[i] = self.id_dict[row_dst]
                i += 1

            self.e = np.rec.array(
                (src_array, dst_array),
                dtype=[('src', self.id_dtype), ('dst', self.id_dtype)])

        elif isinstance(src, str) and isinstance(dst, str):
            assert e[src].isin(self.id_dict).all(), smsg
            assert e[dst].isin(self.id_dict).all(), dmsg

            self.e = np.rec.array(
                (e[src].replace(self.id_dict, inplace=False).to_numpy(),
                 e[dst].replace(self.id_dict, inplace=False).to_numpy()),
                dtype=[('src', self.id_dtype), ('dst', self.id_dtype)])

        else:
            raise ValueError('src and dst can be either both lists or str.')

        # Sort e
        self.sort_ind = np.argsort(self.e)
        self.e = self.e[self.sort_ind]

        # Check that there are no repeated pair in the edge list
        mt._check_unique_edges(self.e)

        # Compute degree (undirected)
        d = mt._compute_degree(self.e, self.num_vertices)
        dtype = 'u' + str(mt._get_num_bytes(np.max(d)))
        self.v = np.rec.array(d.astype(dtype), dtype=[('degree', dtype)])

        # Warn if vertices have no edges
        zero_idx = np.nonzero(d == 0)[0]
        if len(zero_idx) == 1:
            warnings.warn(str(list(self.id_dict.keys())[zero_idx[0]]) +
                          " vertex has no edges.", UserWarning)

        if len(zero_idx) > 1:
            names = []
            for idx in zero_idx:
                names.append(list(self.id_dict.keys())[idx])
            warnings.warn(str(names) + " vertices have no edges.",
                          UserWarning)

    def degree(self, get=False):
        """ Compute the undirected degree sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'degree' in self.v.dtype.names:
            degree = self.v.degree
        else:
            degree = mt._compute_degree(self.e, self.num_vertices)
            dtype = 'u' + str(mt._get_num_bytes(np.max(degree)))
            self.v = append_fields(self.v, 'degree', degree.astype(dtype),
                                   dtypes=dtype)

        if get:
            return degree

    def out_degree(self, get=False):
        """ Compute the out degree sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'out_degree' in self.v.dtype.names:
            d_out = self.v.out_degree
        else:
            d_out, d_in = mt._compute_in_out_degrees(self.e,
                                                     self.num_vertices)
            dtype = 'u' + str(mt._get_num_bytes(max(np.max(d_out),
                                                    np.max(d_in))))
            self.v = append_fields(self.v,
                                   ['out_degree', 'in_degree'],
                                   (d_out.astype(dtype), d_in.astype(dtype)),
                                   dtypes=[dtype, dtype])

        if get:
            return d_out

    def in_degree(self, get=False):
        """ Compute the out degree sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'in_degree' in self.v.dtype.names:
            d_in = self.v.in_degree
        else:
            d_out, d_in = mt._compute_in_out_degrees(self.e,
                                                     self.num_vertices)
            dtype = 'u' + str(mt._get_num_bytes(max(np.max(d_out),
                                                    np.max(d_in))))
            self.v = append_fields(self.v,
                                   ['out_degree', 'in_degree'],
                                   (d_out.astype(dtype), d_in.astype(dtype)),
                                   dtypes=[dtype, dtype])

        if get:
            return d_in


class WeightedGraph(DirectedGraph):
    """ General class for directed graphs with weighted edges.

    Attributes
    ----------
    total_weight: numpy.float64
        sum of all the weights of the edges

    Methods
    -------
    strength:
        compute the undirected strength sequence
    out_strength:
        compute the out strength sequence
    in_strength:
        compute the in strength sequence
    """
    def __init__(self, v, e, v_id, src, dst, weight):
        """Return a sGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str (list of str not yet supported)
            identifier column for the source vertex
        dst: str (list of str not yet supported)
            identifier column for the destination vertex
        weight:
            identifier column for the weight of the edges

        Returns
        -------
        sGraph
            the graph object
        """
        super().__init__(v, e, v_id=v_id, src=src, dst=dst)

        # Convert weights to float64 for computations
        self.e = append_fields(
            self.e,
            'weight',
            e[weight].to_numpy().astype(np.float64)[self.sort_ind],
            dtypes=np.float64)

        # Ensure all weights are positive
        msg = 'Zero or negative edge weights are not supported.'
        assert np.all(self.e.weight > 0), msg

        # Compute total weight
        self.total_weight = np.sum(self.e.weight)

    def strength(self, get=False):
        """ Compute the undirected strength sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'strength' in self.v.dtype.names:
            strength = self.v.strength
        else:
            strength = mt._compute_strength(self.e, self.num_vertices)
            self.v = append_fields(self.v, 'strength', strength,
                                   dtypes=np.float64)

        if get:
            return strength

    def out_strength(self, get=False):
        """ Compute the out strength sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'out_strength' in self.v.dtype.names:
            s_out = self.v.out_strength
        else:
            s_out, s_in = mt._compute_in_out_strengths(self.e,
                                                       self.num_vertices)
            self.v = append_fields(self.v,
                                   ['out_strength', 'in_strength'],
                                   (s_out, s_in),
                                   dtypes=[np.float64, np.float64])

        if get:
            return s_out

    def in_strength(self, get=False):
        """ Compute the out strength sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'in_strength' in self.v.dtype.names:
            s_in = self.v.in_strength
        else:
            s_out, s_in = mt._compute_in_out_strengths(self.e,
                                                       self.num_vertices)

            self.v = append_fields(self.v,
                                   ['out_strength', 'in_strength'],
                                   (s_out, s_in),
                                   dtypes=[np.float64, np.float64])

        if get:
            return s_in


class EdgelabelGraph(sGraph):
    """ General class for directed graphs with labelled edges.

    Attributes
    ----------
    num_vertices: int
        number of vertices in the graph
    num_edges: int
        number of edges in the graph
    v: numpy.rec.array
        array containing the computed properties of the vertices
    e: numpy.rec.array
        array containing the edge list in a condensed format
    id_dict: dict
        dictionary to convert original identifiers to positions in v
    id_type: numpy.dtype
        type of the id (e.g. np.uint16)
    sort_ind: numpy.array
        the index used for sorting e

    Methods
    -------
    degree:
        compute the undirected degree sequence
    out_degree:
        compute the out degree sequence
    in_degree:
        compute the in degree sequence
    """
    def __init__(self, v, e, v_id, src, dst, edge_label):
        """Return a sGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str (list of str not yet supported)
            identifier column for the source vertex
        dst: str (list of str not yet supported)
            identifier column for the destination vertex
        edge_label:
            identifier column for label of the edges

        Returns
        -------
        sGraph
            the graph object
        """
        assert isinstance(v, pd.DataFrame), 'Only dataframe input supported.'
        assert isinstance(e, pd.DataFrame), 'Only dataframe input supported.'

        # Determine size of indices
        self.num_vertices = len(v.index)
        num_bytes = mt._get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype('u' + str(num_bytes))

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        self.id_dict = mt._generate_id_dict(v, v_id)

        # Check that no vertex id in e is not present in v
        assert e[src].isin(self.id_dict).all(), ('Some source vertices are'
                                                 ' not in v.')
        assert e[dst].isin(self.id_dict).all(), ('Some destination vertices'
                                                 ' are not in v.')

        # Get dictionary of label to numeric internal label
        self.label_dict = mt._generate_label_dict(e, edge_label)

        # # Generate optimized edge list and sort it
        # self.e = np.rec.array(
        #     (e[src].replace(self.id_dict, inplace=False).to_numpy(),
        #      e[dst].replace(self.id_dict, inplace=False).to_numpy()),
        #     dtype=[('src', self.id_dtype), ('dst', self.id_dtype)]
        #     )
        # self.sort_ind = np.argsort(self.e)
        # self.e = self.e[self.sort_ind]

        # # Check that there are no repeated pair in the edge list
        # mt._check_unique_edges(self.e)
        # self.num_edges = len(self.e)

        # # Compute degree (undirected)
        # d = mt._compute_degree(self.e, self.num_vertices)
        # dtype = 'u' + str(mt._get_num_bytes(np.max(d)))
        # self.v = np.rec.array(d.astype(dtype), dtype=[('degree', dtype)])

        # # Warn if vertices have no edges
        # zero_idx = np.nonzero(d == 0)[0]
        # if len(zero_idx) == 1:
        #     warnings.warn(str(self.id_list[zero_idx[0]]) +
        #                   " vertex has no edges.", UserWarning)

        # if len(zero_idx) > 1:
        #     names = []
        #     for idx in zero_idx:
        #         names.append(self.id_list[idx])
        #     warnings.warn(str(names) + " vertices have no edges.",
        #                   UserWarning)


class WeightedEdgelabelGraph(EdgelabelGraph, WeightedGraph):
    """ General class for directed graphs with labelled and weighted edges.

    Attributes
    ----------

    Methods
    -------
    """
    def __init__(self, v, e, v_id, src, dst, weight, edge_label):
        """Return a sGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        id_col: str or list of str
            specifies which column uniquely identifies a vertex
        src_col: str (list of str not yet supported)
            identifier column for the source vertex
        dst_col: str (list of str not yet supported)
            identifier column for the destination vertex

        Returns
        -------
        Graph
            the graph object
        """


class GraphModel():
    """ General class for graph models. """
    pass


class FitnessModel(GraphModel):
    pass


class StripeFitnessModel(GraphModel):
    """ A generalized fitness model that allows for vector strength sequences.

    This model allows to take into account labels of the edges and include
    this information as part of the model. The strength sequence is therefore
    now subdivided in strength per label. Two quantities can be preserved by
    the ensemble: either the total number of edges, or the number of edges per
    label.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength sequence
    in_strength: np.ndarray
        the in strength sequence
    num_edges: int (or np.ndarray)
        the total number of edges (per label)
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
                    lambda x: mt.expected_links_stripe_mult_z(
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
                    lambda x: mt.expected_links_stripe_one_z(
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
            return mt.prob_array_stripe_mult_z(self.out_strength,
                                                    self.in_strength,
                                                    self.z,
                                                    self.num_nodes,
                                                    self.num_labels)
        else:
            return mt.prob_array_stripe_one_z(self.out_strength,
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
            return mt.prob_matrix_stripe_mult_z(self.out_strength,
                                                     self.in_strength,
                                                     self.z,
                                                     self.num_nodes)
        else:
            return mt.prob_matrix_stripe_one_z(self.out_strength,
                                                    self.in_strength,
                                                    self.z,
                                                    self.num_nodes)


class BlockFitnessModel(GraphModel):
    pass
