""" This module defines the classes of graphs that support the creation of
ensembles and the exploration of their properties. 

This version stores all adjacency matrices and two dimensional properties as 
sparse arrays and is suitable for large graphs. 
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
import warnings
import networkx as nx


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
    e: numpy.rec.array
        array containing the edge list in a condensed format
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

        self.adj_mat = sp.coo_array(
            (np.ones(src_array.shape[0], np.uint8), (src_array, dst_array)), 
            shape=(self.num_vertices, self.num_vertices)).tocsr()

        self.num_edges = self.adj_mat.nnz

    def degree(self, recompute=False):
        """ Compute the undirected degree sequence.
        """
        if not hasattr(self, '_degree') or recompute:
            if not hasattr(self, 'sym_adj_mat'):
                self.sym_adj_mat = self.adj_mat + self.adj_mat.T
                self.sym_adj_mat.data = np.ones(self.sym_adj_mat.nnz, np.uint8)
            self._degree = self.sym_adj_mat.sum(axis=0)

        return self._degree

    def degree_by_group(self, recompute=False):
        """ Compute the undirected degree sequence to and from each group.
        """
        if not hasattr(self, '_degree_by_group') or recompute:
            if not hasattr(self, 'sym_adj_mat'):
                self.sym_adj_mat = self.adj_mat + self.adj_mat.T
                self.sym_adj_mat.data = np.ones(self.sym_adj_mat.nnz, np.uint8)
            
            id_arr, grp_arr, cnt_arr = self.count_indices_by_group(
                self.sym_adj_mat.indptr, self.sym_adj_mat.indices, self.groups)
            self._degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), 
                shape=(self.num_vertices, self.num_groups)).tocsr()

        return self._degree_by_group

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
    def count_indices_by_group(indptr, indices, g_arr):
        # Get number of index of first dimension
        N = len(indptr)-1
        
        # Convert indices of matrix using group dict
        groups = [g_arr[x] for x in indices]
        
        # Count group occurrences
        id_arr = []
        grp_arr = []
        cnt_arr = []
        for i in range(N):
            m = indptr[i]
            n = indptr[i+1]
            gcnt = {}
            for x in groups[m:n]:
                if x in gcnt:
                    gcnt[x] = gcnt[x] + 1
                else:
                    gcnt[x] = 1
            id_arr.extend([i]*len(gcnt.keys()))
            grp_arr.extend(gcnt.keys())
            cnt_arr.extend(gcnt.values())
        return id_arr, grp_arr, cnt_arr


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

        # Compute degree (undirected)
        d = self.degree()

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

    def out_degree(self, recompute=False):
        """ Compute the out degree sequence.
        """
        if not hasattr(self, '_out_degree') or recompute:
            self._out_degree = self.adj_mat.sum(axis=1)

        return self._out_degree

    def in_degree(self, recompute=False):
        """ Compute the in degree sequence.
        """
        if not hasattr(self, '_in_degree') or recompute:
            self._in_degree = self.adj_mat.sum(axis=0)

        return self._in_degree

    def out_degree_by_group(self, recompute=False):
        """ Compute the out degree sequence to and from each group.
        """
        
        if not hasattr(self, '_out_degree_by_group') or recompute:
            id_arr, grp_arr, cnt_arr = self.count_indices_by_group(
                self.adj_mat.indptr, self.adj_mat.indices, self.groups)
            self._out_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), 
                shape=(self.num_vertices, self.num_groups)).tocsr()

        return self._out_degree_by_group

    def in_degree_by_group(self, recompute=False):
        """ Compute the in degree sequence to and from each group.
        """
        if not hasattr(self, '_in_degree_by_group') or recompute:
            adj = self.adj_mat.tocsc()
            id_arr, grp_arr, cnt_arr = self.count_indices_by_group(
                adj.indptr, adj.indices, self.groups)
            self._in_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), 
                shape=(self.num_vertices, self.num_groups)).tocsr()

        return self._in_degree_by_group

    def adjacency_matrix(self):
        """ Return the adjacency matrix.
        """
        return self.adj_mat

    def to_networkx(self, original=False):
        # Initialize DiGraph object
        G = nx.DiGraph(self.adj_mat)

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, 'groups'):
            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr)

        return G
