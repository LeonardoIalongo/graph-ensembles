""" This module defines the classes of graphs that support the creation of
ensembles and the exploration of their properties.

This version stores all properties of the graphs as numpy arrays and is 
recommended for small or dense graphs. For large sparse graphs consider the 
sparse module or the spark one. 
"""

import numpy as np
import pandas as pd
from . import lib
import warnings
import networkx as nx
from numba import jit


class Graph():
    """ Generator class for graphs. Returns correct object depending on inputs
    passed.
    """
    def __new__(cls, v, e, **kwargs):
        # Ensure passed arguments are accepted
        allowed_arguments = ['v_id', 'src', 'dst', 'weight',
                             'edge_label', 'v_group']
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

        if 'v_group' in kwargs:
            v_group = kwargs['v_group']
        else:
            v_group = None

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
                                          edge_label=label_col,
                                          v_group=v_group)
            else:
                return WeightedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col,
                                     weight=weight_col,
                                     v_group=v_group)

        else:
            if 'edge_label' in kwargs:
                label_col = kwargs['edge_label']
                return LabelGraph(v,
                                  e,
                                  v_id=id_col,
                                  src=src_col,
                                  dst=dst_col,
                                  edge_label=label_col,
                                  v_group=v_group)
            else:
                return DirectedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col,
                                     v_group=v_group)


class sGraph():
    """ General class for graphs.

    Attributes
    ----------
    num_vertices: int
        number of vertices in the graph
    num_edges: int
        number of distinct directed edges in the graph
    adj_mat: numpy.array
        the adjacency matrix of the graph
    id_dict: dict
        dictionary to convert original identifiers to new position id
    id_dtype: numpy.dtype
        type of the id (e.g. np.uint16)
    num_groups: int  (or None)
        number of vertex groups
    group_dict: dict (or none)
        dictionary to convert v_group columns into numeric ids
    group_dtype: numpy.dtype
        type of the group id

    Methods
    -------

    degree:
        compute the undirected degree sequence
    """
    def __init__(self, v, e, v_id, src, dst, v_group=None):
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
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        sGraph
            the graph object
        """
        assert isinstance(v, pd.DataFrame), 'Only dataframe input supported.'
        assert isinstance(e, pd.DataFrame), 'Only dataframe input supported.'

        # If column names are passed as lists with one elements extract str
        if isinstance(v_id, list) and len(v_id) == 1:
            v_id = v_id[0]
        if isinstance(src, list) and len(src) == 1:
            src = src[0]
        if isinstance(dst, list) and len(dst) == 1:
            dst = dst[0]

        # Determine size of indices
        self.num_vertices = len(v.index)
        num_bytes = self.get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype('u' + str(num_bytes))

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        try:
            if isinstance(v_id, list):
                v['_node'] = list(zip(*[v[x] for x in v_id]))
            else:
                v['_node'] = v[v_id]

            self.id_dict = self.generate_id_dict(v, '_node', check_unique=True)

            # Create index with new id value and sort
            v = v.set_index(v['_node'].apply(
                lambda x: self.id_dict.get(x)).values)
            v = v.sort_index()

        except ValueError as err:
            raise err
        except Exception:
            rep_msg = ('There is at least one repeated id in the vertex '
                       'dataframe.')
            raise Exception(rep_msg)

        # If v_group is given then create dict and array
        if v_group is not None:
            if isinstance(v_group, list) and len(v_group) == 1:
                v_group = v_group[0]

            if isinstance(v_group, list):
                v['_group'] = list(zip(*[v[x] for x in v_group]))
            else:
                v['_group'] = v[v_group]

            self.group_dict = self.generate_id_dict(v, '_group')
            self.num_groups = len(self.group_dict)
            num_bytes = self.get_num_bytes(self.num_groups)
            self.group_dtype = np.dtype('u' + str(num_bytes))
            self.groups = v['_group'].apply(
                lambda x: self.group_dict.get(x)
                ).values.astype(self.group_dtype)

        # Check that no vertex id in e is not present in v
        # and generate optimized edge list
        if isinstance(src, list) and isinstance(dst, list):
            e['_src'] = list(zip(*[e[x] for x in src]))
            e['_dst'] = list(zip(*[e[x] for x in dst]))
        elif isinstance(src, str) and isinstance(dst, str):
            e['_src'] = e[src]
            e['_dst'] = e[dst]
        else:
            raise ValueError('src and dst can be either both lists or str.')

        msg = 'Some vertices in e are not in v.'
        try:
            src_array = e['_src'].apply(lambda x: self.id_dict.get(x)).values
            src_array = src_array.astype(self.id_dtype)
            dst_array = e['_dst'].apply(lambda x: self.id_dict.get(x)).values
            dst_array = dst_array.astype(self.id_dtype)
        except KeyError:
            raise Exception(msg)
        except Exception:
            raise Exception()

        self.adj_mat = np.zeros((self.num_vertices, self.num_vertices), 
                                dtype=np.uint8)
        self.adj_mat[src_array, dst_array] = 1
        self.num_edges = self.adj_mat.sum()

    def get_degree(self, recompute=False):
        """ Compute the undirected degree sequence.
        """
        if not hasattr(self, 'degree') or recompute:
            if not hasattr(self, 'sym_adj_mat'):
                self.sym_adj_mat = self.adj_mat + self.adj_mat.T
                self.sym_adj_mat[self.sym_adj_mat != 0] = 1
            self.degree = self.sym_adj_mat.sum(axis=0)

        return self.degree

    def get_degree_by_group(self, recompute=False):
        """ Compute the undirected degree sequence to and from each group.
        """
        if not hasattr(self, 'degree_by_group') or recompute:
            if not hasattr(self, 'sym_adj_mat'):
                self.sym_adj_mat = self.adj_mat + self.adj_mat.T
                self.sym_adj_mat[self.sym_adj_mat != 0] = 1
            
            self.count_indices_by_group(self.sym_adj_mat, self.groups)

        return self.degree_by_group

    @staticmethod
    def get_num_bytes(num_items):
        """ Determine the number of bytes needed for storing ids for num_items.
        """
        return int(max(2**np.ceil(np.log2(np.log2(num_items + 1)/8)), 1))

    @staticmethod
    def generate_id_dict(df, id_col, check_unique=False):
        """ Return id dictionary for given dataframe columns.
        """
        id_dict = {}

        if check_unique:
            ids, counts = np.unique(df[id_col], return_counts=check_unique)
            assert np.all(counts == 1)
        else:
            ids = np.unique(df[id_col])

        for i, x in enumerate(ids):
            id_dict[x] = i

        return id_dict

    @staticmethod
    @jit(nopython=True)
    def count_indices_by_group(mat, g_arr):
        # Initialize empty result
        M = g_arr.max()
        res = np.zeros((mat.shape[0], M), dtype=np.int64)
        
        # Count group occurrences
        for i in range(M):
            res[:, i] = mat[:, np.where(g_arr == i)[0]].sum(axis=1)
            
        return res


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

    def __init__(self, v, e, v_id, src, dst, v_group=None):
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
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        DirectedGraph
            the graph object
        """
        super().__init__(v, e, v_id=v_id, src=src, dst=dst, v_group=v_group)

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
            dtype = 'u' + str(self.get_num_bytes(max(np.max(d_out),
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
            dtype = 'u' + str(self.get_num_bytes(max(np.max(d_out),
                                                   np.max(d_in))))
            self.v = append_fields(self.v,
                                   ['out_degree', 'in_degree'],
                                   (d_out.astype(dtype), d_in.astype(dtype)),
                                   dtypes=[dtype, dtype])

        if get:
            return d_in

    def out_degree_by_group(self, get=False):
        """ Compute the out degree sequence to and from each group.

        If get is true it returns the array, else it adds the result to gv.
        """
        if not hasattr(self, 'gv'):
            raise Exception('Graph object does not contain group info.')

        if not hasattr(self.gv, 'out_degree'):
            d_out, d_in, dout_dict, din_dict = \
                mt.compute_in_out_degree_by_group(self.e, self.v.group)
            dtype = 'u' + str(self.get_num_bytes(max(np.max(d_out[:, 2]),
                                                   np.max(d_in[:, 2]))))
            self.gv.out_degree = d_out.view(
                type=np.recarray,
                dtype=[('id', 'u8'), ('group', 'u8'), ('value', 'u8')]
                ).reshape((d_out.shape[0],)).astype(
                [('id', self.id_dtype),
                 ('group', self.group_dtype),
                 ('value', dtype)]
                )
            self.gv.in_degree = d_in.view(
                type=np.recarray,
                dtype=[('id', 'u8'), ('group', 'u8'), ('value', 'u8')]
                ).reshape((d_in.shape[0],)).astype(
                [('id', self.id_dtype),
                 ('group', self.group_dtype),
                 ('value', dtype)]
                )
            self.gv.out_degree.sort()
            self.gv.in_degree.sort()
            self.gv.out_degree_dict = dout_dict
            self.gv.in_degree_dict = din_dict

        if get:
            return self.gv.out_degree

    def in_degree_by_group(self, get=False):
        """ Compute the in degree sequence to and from each group.

        If get is true it returns the array, else it adds the result to gv.
        """
        if not hasattr(self, 'gv'):
            raise Exception('Graph object does not contain group info.')

        if not hasattr(self.gv, 'in_degree'):
            self.out_degree_by_group()

        if get:
            return self.gv.in_degree

    def adjacency_matrix(self, kind='csr'):
        """ Return the adjacency matrix as a scipy sparse matrix.
        """
        return lib.to_sparse(
            self.e, (self.num_vertices, self.num_vertices), kind=kind,
            i_col='src', j_col='dst', data_col=np.ones(len(self.e)))

    def to_networkx(self, original=False):
        G = nx.DiGraph()
        if original:
            id_conv = list(self.id_dict.keys())

            if hasattr(self, 'gv'):
                group_conv = list(self.group_dict.keys())
                v_num = mt.id_attr_dict(
                    self.v, id_col='id', attr_cols=['group'])
                v = []
                for row in v_num:
                    v.append((id_conv[row[0]],
                             {'group': group_conv[row[1]['group']]}))
            else:
                v_num = self.v.id
                v = []
                for row in v_num:
                    v.append(id_conv[row])

            e = []
            for row in self.e[['src', 'dst']]:
                e.append((id_conv[row[0]], id_conv[row[1]]))
        else:
            if hasattr(self, 'gv'):
                v = mt.id_attr_dict(self.v, id_col='id', attr_cols=['group'])
            else:
                v = self.v.id

            e = self.e[['src', 'dst']]

        G.add_nodes_from(v)
        G.add_edges_from(e)

        return G


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
    def __init__(self, v, e, v_id, src, dst, weight, v_group=None):
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
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        WeightedGraph
            the graph object
        """
        super().__init__(v, e, v_id=v_id, src=src, dst=dst, v_group=v_group)

        if isinstance(weight, list) and len(weight) == 1:
            weight = weight[0]

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

    def strength_by_group(self, get=False):
        """ Compute the undirected strength sequence to and from each group.

        If get is true it returns the array otherwise it adds the result to v.
        """
        if not hasattr(self, 'gv'):
            raise Exception('Graph object does not contain group info.')

        if not hasattr(self.gv, 'strength'):
            s, s_dict = mt.compute_strength_by_group(self.e, self.v.group)
            self.gv.strength = s.view(
                type=np.recarray,
                dtype=[('id', 'f8'), ('group', 'f8'), ('value', 'f8')]
                ).reshape((s.shape[0],)).astype(
                [('id', self.id_dtype),
                 ('group', self.group_dtype),
                 ('value', 'f8')]
                )
            self.gv.strength.sort()
            self.gv.strength_dict = s_dict

        if get:
            return self.gv.strength

    def out_strength_by_group(self, get=False):
        """ Compute the out strength sequence to and from each group.

        If get is true it returns the array, else it adds the result to gv.
        """
        if not hasattr(self, 'gv'):
            raise Exception('Graph object does not contain group info.')

        if not hasattr(self.gv, 'out_strength'):
            s_out, s_in, sout_dict, sin_dict = \
                mt.compute_in_out_strength_by_group(self.e, self.v.group)
            self.gv.out_strength = s_out.view(
                type=np.recarray,
                dtype=[('id', 'f8'), ('group', 'f8'), ('value', 'f8')]
                ).reshape((s_out.shape[0],)).astype(
                [('id', self.id_dtype),
                 ('group', self.group_dtype),
                 ('value', 'f8')]
                )
            self.gv.in_strength = s_in.view(
                type=np.recarray,
                dtype=[('id', 'f8'), ('group', 'f8'), ('value', 'f8')]
                ).reshape((s_in.shape[0],)).astype(
                [('id', self.id_dtype),
                 ('group', self.group_dtype),
                 ('value', 'f8')]
                )
            self.gv.out_strength.sort()
            self.gv.in_strength.sort()
            self.gv.out_strength_dict = sout_dict
            self.gv.in_strength_dict = sin_dict

        if get:
            return self.gv.out_strength

    def in_strength_by_group(self, get=False):
        """ Compute the in strength sequence to and from each group.

        If get is true it returns the array, else it adds the result to gv.
        """
        if not hasattr(self, 'gv'):
            raise Exception('Graph object does not contain group info.')

        if not hasattr(self.gv, 'in_strength'):
            self.out_strength_by_group()

        if get:
            return self.gv.in_strength

    def adjacency_matrix(self, kind='csr'):
        """ Return the adjacency matrix as a scipy sparse matrix.
        """
        return lib.to_sparse(
            self.e, (self.num_vertices, self.num_vertices), kind=kind,
            i_col='src', j_col='dst', data_col='weight')

    def to_networkx(self, original=False):
        G = nx.DiGraph()
        if original:
            id_conv = list(self.id_dict.keys())

            if hasattr(self, 'gv'):
                group_conv = list(self.group_dict.keys())
                v_num = mt.id_attr_dict(
                    self.v, id_col='id', attr_cols=['group'])
                v = []
                for row in v_num:
                    v.append((id_conv[row[0]],
                             {'group': group_conv[row[1]['group']]}))
            else:
                v_num = self.v.id
                v = []
                for row in v_num:
                    v.append(id_conv[row])

            e = []
            for row in self.e[['src', 'dst', 'weight']]:
                e.append((id_conv[row[0]], id_conv[row[1]], row[2]))
        else:
            if hasattr(self, 'gv'):
                v = mt.id_attr_dict(self.v, id_col='id', attr_cols=['group'])
            else:
                v = self.v.id

            e = self.e[['src', 'dst', 'weight']]

        G.add_nodes_from(v)
        G.add_weighted_edges_from(e)

        return G


class LabelVertexList():
    """ Class to store results of label-vertex properties from LabelGraph.
    """
    pass


class LabelGraph(DirectedGraph):
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
    degree_by_label:
        compute the degree of each vertex by label
    out_degree_by_label:
        compute the out degree of each vertex by label
    in_degree_by_label:
        compute the in degree of each vertex by label
    """
    def __init__(self, v, e, v_id, src, dst, edge_label, v_group=None):
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
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        LabelGraph
            the graph object
        """
        super(DirectedGraph, self).__init__(v, e, v_id=v_id, src=src, dst=dst,
                                            v_group=v_group)

        if isinstance(edge_label, list) and len(edge_label) == 1:
            edge_label = edge_label[0]

        # Get dictionary of label to numeric internal label
        self.label_dict = self.generate_id_dict(e, edge_label)
        self.num_labels = len(self.label_dict)
        num_bytes = self.get_num_bytes(self.num_labels)
        self.label_dtype = np.dtype('u' + str(num_bytes))

        # Convert labels
        if isinstance(edge_label, list):
            n = len(edge_label)
            lbl_array = np.empty(len(self.e), dtype=self.label_dtype)
            i = 0
            for row in e[edge_label].itertuples(index=False):
                lbl_array[i] = self.label_dict[row[0:n]]
                i += 1

        elif isinstance(edge_label, str):
            lbl_array = np.empty(len(self.e), dtype=self.label_dtype)
            i = 0
            for row in e[edge_label]:
                lbl_array[i] = self.label_dict[row]
                i += 1

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
        dtype = 'u' + str(self.get_num_bytes(np.max(ne_label)))
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

    def degree_by_label(self, get=False):
        """ Compute the degree sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'degree'):
            d, d_dict = mt.compute_degree_by_label(self.e)
            dtype = 'u' + str(self.get_num_bytes(np.max(d[:, 2])))
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

            dtype = 'u' + str(self.get_num_bytes(max(np.max(d_out[:, 2]),
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

    def adjacency_matrix(self, kind='csr', compressed=False):
        """ Return the adjacency matrices for each label as a list of scipy
        sparse matrices.
        """
        if compressed:
            e = np.unique(self.e[['src', 'dst']])
            adj = lib.to_sparse(
                    e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col=np.ones(len(e)))
        else:
            adj = []
            for i in range(self.num_labels):
                e = self.e[self.e.label == i]
                adj.append(lib.to_sparse(
                    e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col=np.ones(len(e))))
        return adj

    def to_networkx(self, original=False, compressed=False):
        if not compressed:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()

        if original:
            id_conv = list(self.id_dict.keys())
            if not compressed:
                label_conv = list(self.label_dict.keys())

            if hasattr(self, 'gv'):
                group_conv = list(self.group_dict.keys())
                v_num = mt.id_attr_dict(
                    self.v, id_col='id', attr_cols=['group'])
                v = []
                for row in v_num:
                    v.append((id_conv[row[0]],
                             {'group': group_conv[row[1]['group']]}))
            else:
                v_num = self.v.id
                v = []
                for row in v_num:
                    v.append(id_conv[row])

            e = []
            for row in self.e:
                if not compressed:
                    e.append((id_conv[row.src], id_conv[row.dst],
                             {'label': label_conv[row.label]}))
                else:
                    e.append((id_conv[row[0]], id_conv[row[1]]))
        else:
            if hasattr(self, 'gv'):
                v = mt.id_attr_dict(self.v, id_col='id', attr_cols=['group'])
            else:
                v = self.v.id

            if not compressed:
                e = []
                for row in self.e:
                    e.append((row.src, row.dst, {'label': row.label}))
            else:
                e = self.e[['src', 'dst']]

        G.add_nodes_from(v)
        G.add_edges_from(e)

        return G


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
    def __init__(self, v, e, v_id, src, dst, weight, edge_label, v_group=None):
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
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        sGraph
            the graph object
        """
        LabelGraph.__init__(self, v, e, v_id=v_id, src=src, dst=dst,
                            edge_label=edge_label, v_group=v_group)

        if isinstance(weight, list) and len(weight) == 1:
            weight = weight[0]

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

    def adjacency_matrix(self, kind='csr', compressed=False):
        """ Return the adjacency matrices for each label as a list of scipy
        sparse matrices.
        """
        if compressed:
            adj = lib.to_sparse(
                    self.e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col='weight')
        else:
            adj = []
            for i in range(self.num_labels):
                e = self.e[self.e.label == i]
                adj.append(lib.to_sparse(
                    e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col='weight'))
        return adj

    def to_networkx(self, original=False):
        G = nx.MultiDiGraph()
        if original:
            id_conv = list(self.id_dict.keys())
            label_conv = list(self.label_dict.keys())

            if hasattr(self, 'gv'):
                group_conv = list(self.group_dict.keys())
                v_num = mt.id_attr_dict(
                    self.v, id_col='id', attr_cols=['group'])
                v = []
                for row in v_num:
                    v.append((id_conv[row[0]],
                             {'group': group_conv[row[1]['group']]}))
            else:
                v_num = self.v.id
                v = []
                for row in v_num:
                    v.append(id_conv[row])

            e = []
            for row in self.e:
                e.append((id_conv[row.src], id_conv[row.dst],
                         {'weight': row.weight,
                          'label': label_conv[row.label]}))
        else:
            if hasattr(self, 'gv'):
                v = mt.id_attr_dict(self.v, id_col='id', attr_cols=['group'])
            else:
                v = self.v.id

            e = []
            for row in self.e:
                e.append((row.src, row.dst,
                         {'weight': row.weight,
                          'label': row.label}))

        G.add_nodes_from(v)
        G.add_edges_from(e)

        return G
