""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from numpy.lib.recfunctions import rec_append_fields as append_fields
import pandas as pd
import graph_ensembles.methods as mt
import warnings
from math import exp, log


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
        num_bytes = mt.get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype('u' + str(num_bytes))
        self.v = np.arange(self.num_vertices, dtype=self.id_dtype).view(
            type=np.recarray, dtype=[('id', self.id_dtype)])

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        self.id_dict = mt.generate_id_dict(v, v_id)

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
            degree = mt.compute_degree(self.e, self.num_vertices)
            dtype = 'u' + str(mt.get_num_bytes(np.max(degree)))
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
        mt.check_unique_edges(self.e)

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
            d_out, d_in = mt.compute_in_out_degree(self.e,
                                                   self.num_vertices)
            dtype = 'u' + str(mt.get_num_bytes(max(np.max(d_out),
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
            d_out, d_in = mt.compute_in_out_degree(self.e,
                                                   self.num_vertices)
            dtype = 'u' + str(mt.get_num_bytes(max(np.max(d_out),
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
            strength = mt.compute_strength(self.e, self.num_vertices)
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
            s_out, s_in = mt.compute_in_out_strength(self.e,
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
            s_out, s_in = mt.compute_in_out_strength(self.e,
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
        self.label_dict = mt.generate_label_dict(e, edge_label)
        self.num_labels = len(self.label_dict)
        num_bytes = mt.get_num_bytes(self.num_labels)
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
        mt.check_unique_labelled_edges(self.e)

        # Compute number of edges by label
        ne_label = mt.compute_num_edges_by_label(self.e, self.num_labels)
        dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
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
            d_out, d_in = mt.compute_in_out_degree_labelled(
                self.e, self.num_vertices)
            dtype = 'u' + str(mt.get_num_bytes(max(np.max(d_out),
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
            d_out, d_in = mt.compute_in_out_degree_labelled(
                self.e, self.num_vertices)
            dtype = 'u' + str(mt.get_num_bytes(max(np.max(d_out),
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
            d, d_dict = mt.compute_degree_by_label(self.e)
            dtype = 'u' + str(mt.get_num_bytes(np.max(d[:, 2])))
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
                mt.compute_in_out_degree_by_label(self.e)

            dtype = 'u' + str(mt.get_num_bytes(max(np.max(d_out[:, 2]),
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
        self.total_weight_label = mt.compute_tot_weight_by_label(
            self.e, self.num_labels)

    def strength_by_label(self, get=False):
        """ Compute the out strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'strength'):
            s, s_dict = mt.compute_strength_by_label(self.e)

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
        if not hasattr(self.lv, 'out_strength'):
            s_out, s_in, sout_dict, sin_dict = \
                mt.compute_in_out_strength_by_label(self.e)

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
    edges (per label) only. We assume the graph is directed.

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
        """ Return a RandomGraph ensemble.
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
                        self.num_edges = self.num_edges[0]
                else:
                    self.num_labels = None

            msg = ('Number of edges must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.num_edges), msg
            else:
                try:
                    int(self.num_edges)
                except Exception:
                    assert False, msg

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
                    try:
                        int(self.num_edges)
                    except Exception:
                        assert False, msg

        elif hasattr(self, 'q'):
            msg = ('q must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.q), msg
            else:
                try:
                    int(self.q)
                except Exception:
                    assert False, msg

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
        num_bytes = mt.get_num_bytes(self.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges
        if self.num_labels is None:
            if hasattr(self, 'q'):
                e = mt.random_graph(self.num_vertices, self.p,
                                    self.q, self.discrete_weights)
                e = e.view(type=np.recarray,
                           dtype=[('src', 'f8'),
                                  ('dst', 'f8'),
                                  ('weight', 'f8')]).reshape((e.shape[0],))
                e = e.astype([('src', g.id_dtype),
                              ('dst', g.id_dtype),
                              ('weight', 'f8')])
                g.total_weight = np.sum(e.weight)

            else:
                e = mt.random_graph(self.num_vertices, self.p)
                e = e.view(type=np.recarray,
                           dtype=[('src', 'f8'),
                                  ('dst', 'f8')]).reshape((e.shape[0],))
                e = e.astype([('src', g.id_dtype), ('dst', g.id_dtype)])

            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = len(g.e)

        else:
            if hasattr(self, 'q'):
                e = mt.random_labelgraph(self.num_vertices,
                                         self.num_labels,
                                         self.p,
                                         self.q,
                                         self.discrete_weights)
                e = e.view(type=np.recarray,
                           dtype=[('label', 'f8'),
                                  ('src', 'f8'),
                                  ('dst', 'f8'),
                                  ('weight', 'f8')]).reshape((e.shape[0],))
                g.num_labels = self.num_labels
                num_bytes = mt.get_num_bytes(g.num_labels)
                g.label_dtype = np.dtype('u' + str(num_bytes))

                e = e.astype([('label', g.label_dtype),
                              ('src', g.id_dtype),
                              ('dst', g.id_dtype),
                              ('weight', 'f8')])
                g.total_weight = np.sum(e.weight)
                g.total_weight_label = mt.compute_tot_weight_by_label(
                    e, self.num_labels)

            else:
                e = mt.random_labelgraph(self.num_vertices,
                                         self.num_labels,
                                         self.p)
                e = e.view(type=np.recarray,
                           dtype=[('label', 'f8'),
                                  ('src', 'f8'),
                                  ('dst', 'f8')]).reshape((e.shape[0],))
                g.num_labels = self.num_labels
                num_bytes = mt.get_num_bytes(g.num_labels)
                g.label_dtype = np.dtype('u' + str(num_bytes))

                e = e.astype([('label', g.label_dtype),
                              ('src', g.id_dtype),
                              ('dst', g.id_dtype)])

            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = len(g.e)
            ne_label = mt.compute_num_edges_by_label(g.e, g.num_labels)
            dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
            g.num_edges_label = ne_label.astype(dtype)

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
    num_vertices: int
        the total number of nodes
    num_labels: int
        the total number of labels by which the vector strengths are computed
    z: float or np.ndarray
        the density parameter (or vector of)
    """

    def __init__(self, *args, **kwargs):
        """ Return a VectorFitnessModel for the given graph data.

        The model accepts as arguments either: a WeightedLabelGraph, the
        strength sequences (in and out) and the number of edges (per label),
        or the strength sequences and the z parameter (per label).

        The model accepts the strength sequences as numpy arrays. The first
        column must contain the node index, the second column the label index
        to which the strength refers, and in the third column must have the
        value of the strength for the node label pair. All node label pairs
        not included are assumed zero.

        TO DO: add functionality for single z
        (Note that the number of edges given implicitly determines if the
        quantity preserved is the total number of edges or the number of edges
        per label. Pass only one integer for the first and a numpy array for
        the second. Note that if an array is passed then the index must be the
        same as the one in the strength sequence.)

        """

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], WeightedLabelGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges_label
                self.num_labels = g.num_labels
                self.out_strength = g.out_strength_by_label(get=True)
                self.in_strength = g.in_strength_by_label(get=True)
            else:
                raise ValueError('First argument passed must be a '
                                 'WeightedLabelGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'num_labels',
                             'out_strength', 'in_strength', 'z',
                             'discrete_weights']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)
            else:
                setattr(self, name, kwargs[name])

        # Ensure that all necessary fields have been set
        if not hasattr(self, 'num_vertices'):
            raise ValueError('Number of vertices not set.')

        if not hasattr(self, 'num_labels'):
            raise ValueError('Number of labels not set.')

        if not hasattr(self, 'out_strength'):
            raise ValueError('out_strength not set.')

        if not hasattr(self, 'in_strength'):
            raise ValueError('in_strength not set.')

        if not (hasattr(self, 'num_edges') or
                hasattr(self, 'z')):
            raise ValueError('Either num_edges or z must be set.')

        # Ensure that strengths passed adhere to format
        msg = ("Out strength must be a rec array with columns: "
               "('label', 'id', 'value')")
        assert isinstance(self.out_strength, np.recarray), msg
        for clm in self.out_strength.dtype.names:
            assert clm in ('label', 'id', 'value'), msg

        msg = ("In strength must be a rec array with columns: "
               "('label', 'id', 'value')")
        assert isinstance(self.in_strength, np.recarray), msg
        for clm in self.in_strength.dtype.names:
            assert clm in ('label', 'id', 'value'), msg

        # Ensure that strengths are sorted
        self.out_strength.sort()
        self.in_strength.sort()

        # Ensure that number of constraint matches number of labels
        if hasattr(self, 'num_edges'):
            if isinstance(self.num_edges, np.ndarray):
                msg = ('Number of edges array does not have the number of'
                       ' elements equal to the number of labels.')
                assert len(self.num_edges) == self.num_labels, msg
            else:
                raise ValueError('Single number of edges not yet supported.')
        else:
            if isinstance(self.z, np.ndarray):
                msg = ('The z array does not have the number of'
                       ' elements equal to the number of labels.')
                assert len(self.z) == self.num_labels, msg
            else:
                raise ValueError('Single z not yet supported.')

        # Check that sum of in and out strengths are equal per label
        tot_out = np.zeros((self.num_labels))
        for row in self.out_strength:
            tot_out[row[0]] += row[2]
        tot_in = np.zeros((self.num_labels))
        for row in self.in_strength:
            tot_in[row[0]] += row[2]

        msg = 'Sum of strengths per label do not match.'
        assert np.allclose(tot_out, tot_in, atol=1e-6), msg

        # If z is set computed expected number of edges per label
        if hasattr(self, 'z'):
            self.num_edges = mt.exp_edges_stripe(self.z,
                                                 self.out_strength,
                                                 self.in_strength,
                                                 self.num_labels)

    def fit(self, z0=None, method="newton", tol=1e-8,
            xtol=1e-8, max_iter=100, verbose=False):
        """ Compute the optimal z to match the given number of edges.

        Parameters
        ----------
        z0: float or np.ndarray
            optional initial conditions for z parameters
        method: 'newton' or 'fixed-point'
            selects which method to use for the solver
        tol : float
            tolerance for the exit condition on the norm
        eps : float
            tolerance for the exit condition on difference between two
            iterations
        max_iter : int or float
            maximum number of iteration
        verbose: boolean
            if true print debug info while iterating

        """
        if isinstance(self.num_edges, np.ndarray):
            z = np.empty(self.num_labels, dtype=np.float64)
            self.solver_output = [None]*self.num_labels
            for i in range(self.num_labels):
                s_out = self.out_strength[self.out_strength.label == i]
                s_in = self.in_strength[self.in_strength.label == i]
                num_e = self.num_edges[i]
                if z0 is None:
                    x0 = mt.stripe_newton_init(s_out, s_in, num_e, 2)
                else:
                    if isinstance(z0, np.ndarray):
                        x0 = z0[i]
                    else:
                        raise ValueError('Single z not yet supported.')

                if method == "newton":
                    sol = mt.newton_solver(
                        x0=log(x0),
                        fun=lambda x: mt.f_jac_stripe_single_layer(
                            x, s_out, s_in, num_e),
                        tol=tol,
                        xtol=xtol,
                        max_iter=max_iter,
                        verbose=verbose,
                        full_return=True)
                elif method == "fixed-point":
                    sol = mt.fixed_point_solver(
                        x0=log(x0),
                        fun=lambda x: mt.iterative_stripe_single_layer(
                            x,
                            s_out,
                            s_in,
                            num_e),
                        xtol=xtol,
                        max_iter=max_iter,
                        verbose=verbose,
                        full_return=True)

                else:
                    raise ValueError("The selected method is not valid.")

                # Update results and check convergence
                z[i] = exp(sol.x)
                self.solver_output[i] = sol

                if not sol.converged:
                    self.solver_output[i] = sol
                    if method == 'newton':
                        mod = sol.norm_seq[-1]
                    else:
                        mod = mt.exp_edges_stripe_single_layer(
                            log(z[i]), s_out, s_in) - num_e
                    if sol.max_iter_reached:
                        msg = ('Fit of layer {} '.format(i) + 'did not '
                               'converge: \n solver stopped because it '
                               'reached the max number of iterations. \n'
                               'Final distance from root = {}'.format(mod))
                        warnings.warn(msg, UserWarning)

                    if method == 'newton':
                        if sol.no_change_stop:
                            msg = ('Fit of layer {} '.format(i) + 'did not '
                                   'converge: \n solver stopped because the '
                                   'update of x was smaller than the '
                                   ' tolerance. \n Final distance from'
                                   ' root = {}'.format(mod))
                            warnings.warn(msg, UserWarning)

            # Collate results
            self.z = z
        else:
            raise ValueError('Single z not yet supported.')

    def expected_num_edges(self):
        """ Compute the expected number of edges (per label).
        """
        if hasattr(self, 'z'):
            return mt.exp_edges_stripe(self.z,
                                       self.out_strength,
                                       self.in_strength,
                                       self.num_labels)
        else:
            raise Exception('Model must be fitted before hand.')

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'z'):
            raise Exception('Ensemble has to be fitted before sampling.')

        # Generate uninitialised graph object
        g = WeightedLabelGraph.__new__(WeightedLabelGraph)
        g.lv = LabelVertexList()

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = mt.get_num_bytes(self.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges and extract properties
        e = mt.stripe_sample(self.z, self.out_strength,
                             self.in_strength, self.num_labels)
        e = e.view(type=np.recarray,
                   dtype=[('label', 'f8'),
                          ('src', 'f8'),
                          ('dst', 'f8'),
                          ('weight', 'f8')]).reshape((e.shape[0],))
        g.num_labels = self.num_labels
        num_bytes = mt.get_num_bytes(g.num_labels)
        g.label_dtype = np.dtype('u' + str(num_bytes))
        e = e.astype([('label', g.label_dtype),
                      ('src', g.id_dtype),
                      ('dst', g.id_dtype),
                      ('weight', 'f8')])
        g.sort_ind = np.argsort(e)
        g.e = e[g.sort_ind]
        g.num_edges = len(g.e)
        ne_label = mt.compute_num_edges_by_label(g.e, g.num_labels)
        dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
        g.num_edges_label = ne_label.astype(dtype)
        g.total_weight = np.sum(e.weight)
        g.total_weight_label = mt.compute_tot_weight_by_label(
                g.e, g.num_labels)

        return g


class BlockFitnessModel(GraphEnsemble):
    pass
