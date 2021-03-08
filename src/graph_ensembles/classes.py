""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from numpy.lib.recfunctions import rec_append_fields as append_fields
from numpy.random import default_rng
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
                return WeightedLabelGraph(v,
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
                return LabelGraph(v,
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
    id_dtype: numpy.dtype
        type of the id (e.g. np.uint16)

    Methods
    -------

    degree:
        compute the undirected degree sequence
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
        self.v = np.arange(self.num_vertices, dtype=self.id_dtype).view(
            type=np.recarray, dtype=[('id', self.id_dtype)])

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
                (e[src].replace(self.id_dict,
                                inplace=False).to_numpy(dtype=self.id_dtype),
                 e[dst].replace(self.id_dict,
                                inplace=False).to_numpy(dtype=self.id_dtype)),
                dtype=[('src', self.id_dtype), ('dst', self.id_dtype)])

        else:
            raise ValueError('src and dst can be either both lists or str.')

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


class DirectedGraph(sGraph):
    """ General class for directed graphs.

    Attributes
    ----------
    sort_ind: numpy.array
        the index used for sorting e

    Methods
    -------
    out_degree:
        compute the out degree sequence
    in_degree:
        compute the in degree sequence
    """

    def __init__(self, v, e, v_id, src, dst):
        """Return a DirectedGraph object given vertices and edges.

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
        super().__init__(v, e, v_id=v_id, src=src, dst=dst)

        # Sort e
        self.sort_ind = np.argsort(self.e)
        self.e = self.e[self.sort_ind]

        # Check that there are no repeated pair in the edge list
        mt._check_unique_edges(self.e)

        # Compute degree (undirected)
        self.degree()

        # Warn if vertices have no edges
        zero_idx = np.nonzero(self.v.degree == 0)[0]
        if len(zero_idx) == 1:
            warnings.warn(str(list(self.id_dict.keys())[zero_idx[0]]) +
                          " vertex has no edges.", UserWarning)

        if len(zero_idx) > 1:
            names = []
            for idx in zero_idx:
                names.append(list(self.id_dict.keys())[idx])
            warnings.warn(str(names) + " vertices have no edges.",
                          UserWarning)

    def out_degree(self, get=False):
        """ Compute the out degree sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'out_degree' in self.v.dtype.names:
            d_out = self.v.out_degree
        else:
            d_out, d_in = mt._compute_in_out_degree(self.e,
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
            d_out, d_in = mt._compute_in_out_degree(self.e,
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
        """Return a WeightedGraph object given vertices and edges.

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
            s_out, s_in = mt._compute_in_out_strength(self.e,
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
            s_out, s_in = mt._compute_in_out_strength(self.e,
                                                      self.num_vertices)

            self.v = append_fields(self.v,
                                   ['out_strength', 'in_strength'],
                                   (s_out, s_in),
                                   dtypes=[np.float64, np.float64])

        if get:
            return s_in


class LabelVertexList():
    """ Class to store results of label-vertex properties from LabelGraph.
    """
    pass


class LabelGraph(sGraph):
    """ General class for directed graphs with labelled edges.

    Attributes
    ----------
    lv: numpy.rec.array
        contains all properties of the label-vertex pair
    num_labels: int
        number of distinct edge labels
    num_edges_label: numpy.array
        number of edges by label (in order)
    label_dtype: numpy.dtype
        the data type of the label internal id
    sort_ind: numpy.array
        the index used for sorting e

    Methods
    -------
    out_degree:
        compute the out degree sequence
    in_degree:
        compute the in degree sequence
    degree_by_label:
        compute the degree of each vertex by label
    out_degree_by_label:
        compute the out degree of each vertex by label
    in_degree_by_label:
        compute the in degree of each vertex by label
    """
    def __init__(self, v, e, v_id, src, dst, edge_label):
        """Return a LabelGraph object given vertices and edges.

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
        super().__init__(v, e, v_id=v_id, src=src, dst=dst)

        # Get dictionary of label to numeric internal label
        self.label_dict = mt._generate_label_dict(e, edge_label)
        self.num_labels = len(self.label_dict)
        num_bytes = mt._get_num_bytes(self.num_labels)
        self.label_dtype = np.dtype('u' + str(num_bytes))

        # Convert labels
        if isinstance(edge_label, list):
            n = len(edge_label)
            lbl_array = np.empty(self.num_edges, dtype=self.label_dtype)
            i = 0
            for row in e[edge_label].itertuples(index=False):
                lbl_array[i] = self.label_dict[row[0:n]]
                i += 1

        elif isinstance(edge_label, str):
            lbl_array = e[edge_label].replace(
                self.label_dict, inplace=False).to_numpy()

        else:
            raise ValueError('edge_label can be either a list or a str.')

        # Add labels to e
        self.e = append_fields(self.e, 'label', lbl_array,
                               dtypes=self.label_dtype)

        # Put label at the beginning and sort
        self.e = self.e[['label', 'src', 'dst']]
        self.sort_ind = np.argsort(self.e)
        self.e = self.e[self.sort_ind]

        # Check that there are no repeated pair in the edge list
        mt._check_unique_labelled_edges(self.e)

        # Compute number of edges by label
        ne_label = mt._compute_num_edges_by_label(self.e, self.num_labels)
        dtype = 'u' + str(mt._get_num_bytes(np.max(ne_label)))
        self.num_edges_label = ne_label.astype(dtype)

        # Compute degree (undirected)
        self.degree()

        # Warn if vertices have no edges
        zero_idx = np.nonzero(self.v.degree == 0)[0]
        if len(zero_idx) == 1:
            warnings.warn(str(list(self.id_dict.keys())[zero_idx[0]]) +
                          " vertex has no edges.", UserWarning)

        if len(zero_idx) > 1:
            names = []
            for idx in zero_idx:
                names.append(list(self.id_dict.keys())[idx])
            warnings.warn(str(names) + " vertices have no edges.",
                          UserWarning)

        # Create lv property
        self.lv = LabelVertexList()

    def out_degree(self, get=False):
        """ Compute the out degree sequence.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if 'out_degree' in self.v.dtype.names:
            d_out = self.v.out_degree
        else:
            d_out, d_in = mt._compute_in_out_degree_labelled(
                self.e, self.num_vertices)
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
            d_out, d_in = mt._compute_in_out_degree_labelled(
                self.e, self.num_vertices)
            dtype = 'u' + str(mt._get_num_bytes(max(np.max(d_out),
                                                    np.max(d_in))))
            self.v = append_fields(self.v,
                                   ['out_degree', 'in_degree'],
                                   (d_out.astype(dtype), d_in.astype(dtype)),
                                   dtypes=[dtype, dtype])

        if get:
            return d_in

    def degree_by_label(self, get=False):
        """ Compute the degree sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'degree'):
            d, d_dict = mt._compute_degree_by_label(self.e)
            dtype = 'u' + str(mt._get_num_bytes(np.max(d[:, 2])))
            self.lv.degree = d.view(
                type=np.recarray,
                dtype=[('label', 'u8'), ('id', 'u8'), ('value', 'u8')]
                ).reshape((d.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', dtype)]
                )

            self.lv.degree.sort()
            self.lv.degree_dict = d_dict

        if get:
            return self.lv.degree

    def out_degree_by_label(self, get=False):
        """ Compute the out degree sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'out_degree'):
            d_out, d_in, dout_dict, din_dict = \
                mt._compute_in_out_degree_by_label(self.e)

            dtype = 'u' + str(mt._get_num_bytes(max(np.max(d_out[:, 2]),
                                                    np.max(d_in[:, 2]))))
            self.lv.out_degree = d_out.view(
                type=np.recarray,
                dtype=[('label', 'u8'), ('id', 'u8'), ('value', 'u8')]
                ).reshape((d_out.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', dtype)]
                )
            self.lv.in_degree = d_in.view(
                type=np.recarray,
                dtype=[('label', 'u8'), ('id', 'u8'), ('value', 'u8')]
                ).reshape((d_in.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', dtype)]
                )

            self.lv.out_degree.sort()
            self.lv.in_degree.sort()
            self.lv.out_degree_dict = dout_dict
            self.lv.in_degree_dict = din_dict

        if get:
            return self.lv.out_degree

    def in_degree_by_label(self, get=False):
        """ Compute the in degree sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'in_degree'):
            self.out_degree_by_label()

        if get:
            return self.lv.in_degree


class WeightedLabelGraph(WeightedGraph, LabelGraph):
    """ General class for directed graphs with labelled and weighted edges.

    Attributes
    ----------
    total_weight_label: numpy.array
        sum of all the weights of the edges by label

    Methods
    -------
    strength_by_label:
        compute the strength of each vertex by label
    out_strength_by_label:
        compute the out strength of each vertex by label
    in_strength_by_label:
        compute the in strength of each vertex by label
    """
    def __init__(self, v, e, v_id, src, dst, weight, edge_label):
        """Return a WeightedLabelGraph object given vertices and edges.

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
        edge_label:
            identifier column for label of the edges

        Returns
        -------
        sGraph
            the graph object
        """
        LabelGraph.__init__(self, v, e, v_id=v_id, src=src, dst=dst,
                            edge_label=edge_label)

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

        # Compute total weight by label
        self.total_weight_label = mt._compute_tot_weight_by_label(
            self.e, self.num_labels)

    def strength_by_label(self, get=False):
        """ Compute the out strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'strength'):
            s, s_dict = mt._compute_strength_by_label(self.e)

            self.lv.strength = s.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )

            self.lv.strength.sort()
            self.lv.strength_dict = s_dict

        if get:
            return self.lv.strength

    def out_strength_by_label(self, get=False):
        """ Compute the out strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'out_degree'):
            s_out, s_in, sout_dict, sin_dict = \
                mt._compute_in_out_strength_by_label(self.e)

            self.lv.out_strength = s_out.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s_out.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )
            self.lv.in_strength = s_in.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s_in.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )

            self.lv.out_strength.sort()
            self.lv.in_strength.sort()
            self.lv.out_strength_dict = sout_dict
            self.lv.in_strength_dict = sin_dict

        if get:
            return self.lv.out_strength

    def in_strength_by_label(self, get=False):
        """ Compute the in strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'in_strength'):
            self.out_strength_by_label()

        if get:
            return self.lv.in_strength


class GraphEnsemble():
    """ General class for graph ensembles.

    All ensembles can be defined in three ways:

    1) From a suitable Graph object: we can think this as a randomization of
    the observed graph. The conserved quantities and relevant vertex
    attributes are computed on the original graph to initialise the ensemble.
    It is then possible to fit the model parameters in order to get a
    probability distribution over all graphs from which to sample.

    2) From conserved quantities and relevant vertex attributes directly: in
    the case we do not have a reference graph but we do know what properties
    we want the ensemble to hold, we can directly use those properties to
    initialise the model. Once this step is completed we can similarly fit the
    parameters and sample from the ensemble.

    3) Fully specifying all model parameters: a final possibility is to
    initialise the model by giving it the list of parameters it needs in order
    to define the probability distribution over graphs. In this case we do not
    need to fit the model and the value of the conserved quantities over the
    ensemble will depend on the parameters passed to the model rather than
    vice versa.

    What these three possibilities entail will depend on the specifics of the
    model.

    Note that if keyword arguments are passed together with a Graph, then the
    arguments overwrite the graph property. This allows for easier definition
    of the ensemble for example when we want to modify one aspect of the
    reference graph but not all (e.g. only the density, but keeping strengths
    the same).

    """
    pass


class RandomGraph(GraphEnsemble):
    """ The simplest graph ensemble defined by conserving the total number of
    links (per label) only. We assume the graph is directed.

    If it is initialized with a LabelGraph or with a number of labels, then
    the edges will be labelled.

    Attributes
    ----------
    num_vertices: int
        the total number of vertices
    num_labels: int or None
        the total number of labels by which the vector strengths are computed
    num_edges: float (or np.ndarray)
        the total number of edges (per label)
    total_weight: float (or np.ndarray)
        the sum of all edges weights (per label)
    p: float or np.ndarray
        the probability of each link (by label)
    q: float or np.ndarray
        the parameter defining the probability distribution of weights
    discrete_weights: boolean
        the flag determining if the distribution of weights is discrete or
        continuous
    """

    def __init__(self, *args, **kwargs):
        """ Initialize a RandomGraph ensemble.
        """

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], Graph):
                g = args[0]
                self.num_vertices = g.num_vertices
                if isinstance(g, LabelGraph):
                    self.num_edges = g.num_edges_label
                    self.num_labels = g.num_labels
                else:
                    self.num_edges = g.num_edges
                    self.num_labels = None
            else:
                ValueError('First argument passed must be a Graph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'num_labels',
                             'total_weight', 'p', 'q', 'discrete_weights']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)
            else:
                setattr(self, name, kwargs[name])

        # Check that all necessary attributes have been passed
        if not hasattr(self, 'num_vertices'):
            raise ValueError('Number of vertices not set.')

        if hasattr(self, 'p'):
            if hasattr(self, 'num_edges'):
                raise ValueError('Either p or num_edges can be set not both.')
            else:
                if not hasattr(self, 'num_labels'):
                    if isinstance(self.p, np.ndarray):
                        if len(self.p) > 1:
                            self.num_labels = len(self.p)
                        else:
                            self.num_labels = None
                    else:
                        self.num_labels = None

            msg = ('Number of p parameters must be the same as number'
                   ' of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.p), msg
            else:
                assert isinstance(self.p, (int, float)), msg
            self.num_edges = self.get_num_edges()

        else:
            if not hasattr(self, 'num_edges'):
                raise ValueError('Neither p nor num_edges have been set.')

            if not hasattr(self, 'num_labels'):
                if isinstance(self.num_edges, np.ndarray):
                    if len(self.num_edges) > 1:
                        self.num_labels = len(self.num_edges)
                    else:
                        self.num_labels = None
                else:
                    self.num_labels = None

            msg = ('Number of edges must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.num_edges), msg
            else:
                assert isinstance(self.num_edges, (int, float)), msg

        # Check if weight information is present
        if not hasattr(self, 'discrete_weights') and (
         hasattr(self, 'q') or hasattr(self, 'total_weight')):
            self.discrete_weights = False

        if hasattr(self, 'total_weight'):
            if hasattr(self, 'q'):
                msg = 'Either total_weight or q can be set not both.'
                raise Exception(msg)
            else:
                msg = ('total_weight must be a vector with length equal to '
                       'the number of labels.')
                if self.num_labels is not None:
                    assert self.num_labels == len(self.total_weight), msg
                else:
                    assert isinstance(self.total_weight, (int, float)), msg

        elif hasattr(self, 'q'):
            msg = ('q must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.q), msg
            else:
                assert isinstance(self.q, (int, float)), msg

            self.total_weight = self.get_total_weight()

    def fit(self):
        """ Fit the parameter p and q to the number of edges and total weight.
        """
        self.p = self.num_edges/(self.num_vertices*(self.num_vertices - 1))

        if hasattr(self, 'total_weight'):
            if self.discrete_weights:
                self.q = 1 - self.num_edges/self.total_weight
            else:
                self.q = self.num_edges/self.total_weight

    def get_num_edges(self):
        """ Compute the expected number of edges (per label) given p.
        """
        return self.p*self.num_vertices*(self.num_vertices - 1)

    def get_total_weight(self):
        """ Compute the expected total weight (per label) given q.
        """
        if self.discrete_weights:
            return self.num_edges/(1 - self.q)
        else:
            return self.num_edges/self.q

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'p'):
            raise Exception('Ensemble has to be fitted before sampling.')

        # Generate uninitialised graph object
        if self.num_labels is None:
            if hasattr(self, 'q'):
                g = WeightedGraph.__new__(WeightedGraph)
            else:
                g = DirectedGraph.__new__(DirectedGraph)
        else:
            if hasattr(self, 'q'):
                g = WeightedLabelGraph.__new__(WeightedLabelGraph)
            else:
                g = LabelGraph.__new__(LabelGraph)
            g.lv = LabelVertexList()

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = mt._get_num_bytes(self.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges
        if self.num_labels is None:
            e = mt._random_graph(self.num_vertices, self.p)
            e = e.view(type=np.recarray,
                       dtype=[('src', 'u8'),
                              ('dst', 'u8')]).reshape((e.shape[0],))
            e = e.astype([('src', g.id_dtype), ('dst', g.id_dtype)])
            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = len(g.e)
        else:
            e = mt._random_labelgraph(self.num_vertices,
                                      self.num_labels,
                                      self.p)
            e = e.view(type=np.recarray,
                       dtype=[('label', 'u8'),
                              ('src', 'u8'),
                              ('dst', 'u8')]).reshape((e.shape[0],))
            g.num_labels = self.num_labels
            num_bytes = mt._get_num_bytes(g.num_labels)
            g.label_dtype = np.dtype('u' + str(num_bytes))

            e = e.astype([('label', g.label_dtype),
                          ('src', g.id_dtype),
                          ('dst', g.id_dtype)])
            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = len(g.e)
            ne_label = mt._compute_num_edges_by_label(g.e, g.num_labels)
            dtype = 'u' + str(mt._get_num_bytes(np.max(ne_label)))
            g.num_edges_label = ne_label.astype(dtype)

        # Add weights if available
        if hasattr(self, 'q'):
            rnd = default_rng()
            if self.num_labels is None:
                if self.discrete_weights:
                    weights = rnd.geometric(1 - self.q, g.num_edges)
                else:
                    weights = rnd.exponential(1/self.q, g.num_edges)
            else:
                weights = np.empty(g.num_edges, dtype=np.float64)
                start = 0
                for i in range(self.num_labels):
                    end = start + g.num_edges_label[i]
                    if self.discrete_weights:
                        w = rnd.geometric(1-self.q[i], g.num_edges_label[i])
                    else:
                        w = rnd.exponential(1/self.q[i], g.num_edges_label[i])

                    weights[start:end] = w
                    start = end

            g.e = append_fields(g.e,
                                'weight',
                                weights)

            g.total_weight = np.sum(weights)

            if self.num_labels is not None:
                g.total_weight_label = mt._compute_tot_weight_by_label(
                    g.e, g.num_labels)

        return g


class FitnessModel(GraphEnsemble):
    pass


class StripeFitnessModel(GraphEnsemble):
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


class BlockFitnessModel(GraphEnsemble):
    pass
