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


class Graph:
    """General class for undirected graphs.

    Note that edges can be weighted or not. If they are not, the strengths
    will be equal to the degrees. The class does not check for the uniqueness
    of the links definitions. If a link is provided multiple times with
    weights, they will be summed. This also applies to the case where both the
    (i, j, weight) and symmetric (j, i, weight) are provided.

    Attributes
    ----------
    num_vertices: int
        Number of vertices in the graph.
    adj: numpy.array
        The adjacency matrix of the graph.
    id_dict: dict
        Dictionary to convert original identifiers to new position id.
    id_dtype: numpy.dtype
        Type of the id (e.g. np.uint16).
    num_groups: int  (or None)
        Number of vertex groups.
    group_dict: dict (or none)
        Dictionary to convert v_group columns into numeric ids.
    group_dtype: numpy.dtype
        Type of the group id.
    groups: numpy.array
        Array with the group each node belongs to.
    weighted: bool
        Is true if the edges have an associated weight.

    Methods
    -------
    num_edges:
        Compute the number of edges in the graph.
    total_weight:
        Compute the total sum of the weights of the edges.
    degree:
        Compute the undirected degree sequence.
    degree_by_group:
        Compute the undirected degree sequence by group as a 2D array.
    strength:
        Compute the total strength sequence.
    strength_by_group:
        Compute the total strength sequence by group as a 2D array.
    average_nn_property:
        Compute the average nearest neighbour property of each node.
    average_nn_degree:
        Compute the average nearest neighbour degree of each node.
    adjacency_matrix:
        Return the adjacency matrix of the graph.
    to_networkx:
        Return a Networkx equivalent.
    """

    def __init__(self, v, e, v_id, src, dst, weight=None, v_group=None):
        """Return a Graph object given vertices and edges.

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
        weight: str or None
            identifier column for the weight of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        Graph
            the graph object
        """
        assert isinstance(v, pd.DataFrame), "Only dataframe input supported."
        assert isinstance(e, pd.DataFrame), "Only dataframe input supported."

        # If column names are passed as lists with one elements extract str
        if isinstance(v_id, list) and len(v_id) == 1:
            v_id = v_id[0]
        if isinstance(src, list) and len(src) == 1:
            src = src[0]
        if isinstance(dst, list) and len(dst) == 1:
            dst = dst[0]
        if isinstance(v_group, list) and len(v_group) == 1:
            v_group = v_group[0]
        if isinstance(weight, list) and len(weight) == 1:
            weight = weight[0]

        # Determine size of indices
        self.num_vertices = len(v.index)
        num_bytes = self.get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype("u" + str(num_bytes))

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        try:
            if isinstance(v_id, list):
                v["_node"] = list(zip(*[v[x] for x in v_id]))
            else:
                v["_node"] = v[v_id]

            self.id_dict = self.generate_id_dict(v, "_node", check_unique=True)

            # Create index with new id value and sort
            v = v.set_index(v["_node"].map(lambda x: self.id_dict.get(x)).values)
            v = v.sort_index()

        except ValueError as err:
            raise err
        except Exception:
            rep_msg = "There is at least one repeated id in the vertex " "dataframe."
            raise Exception(rep_msg)

        # If v_group is given then create dict and array
        if v_group is not None:
            if isinstance(v_group, list):
                v["_group"] = list(zip(*[v[x] for x in v_group]))
            else:
                v["_group"] = v[v_group]

            self.group_dict = self.generate_id_dict(v, "_group")
            self.num_groups = len(self.group_dict)
            num_bytes = self.get_num_bytes(self.num_groups)
            self.group_dtype = np.dtype("u" + str(num_bytes))
            self.groups = (
                v["_group"]
                .map(lambda x: self.group_dict.get(x))
                .values.astype(self.group_dtype)
            )

        # Check that no vertex id in e is not present in v
        # and generate optimized edge list
        if isinstance(src, list) and isinstance(dst, list):
            e["_src"] = list(zip(*[e[x] for x in src]))
            e["_dst"] = list(zip(*[e[x] for x in dst]))
        elif isinstance(src, str) and isinstance(dst, str):
            e["_src"] = e[src]
            e["_dst"] = e[dst]
        else:
            raise ValueError("src and dst can be either both lists or str.")

        msg = "Some vertices in e are not in v."
        e["_src"] = e["_src"].map(lambda x: self.id_dict.get(x))
        e["_dst"] = e["_dst"].map(lambda x: self.id_dict.get(x))

        # Check for nans
        assert not (np.any(np.isnan(e["_src"])) or np.any(np.isnan(e["_dst"]))), msg

        # Extract values
        src_array = e["_src"].values.astype(self.id_dtype)
        dst_array = e["_dst"].values.astype(self.id_dtype)

        # Construct adjacency matrix
        if weight is not None:
            self.weighted = True
            val_array = e[weight].values

            # Ensure all weights are positive
            msg = "Zero or negative edge weights are not supported."
            assert np.all(val_array > 0), msg

            self.adj = sp.coo_array(
                (val_array, (src_array, dst_array)),
                shape=(self.num_vertices, self.num_vertices),
            ).tocsr()
        else:
            self.weighted = False
            self.adj = sp.coo_array(
                (np.ones(src_array.shape[0], bool), (src_array, dst_array)),
                shape=(self.num_vertices, self.num_vertices),
            ).tocsr()

        # Compute undirected degree
        adj = self.adj != 0
        adj = adj + adj.T
        d = adj.sum(axis=1)

        # Warn if vertices have no edges
        zero_idx = np.nonzero(d == 0)[0]
        if len(zero_idx) == 1:
            warnings.warn(
                str(list(self.id_dict.keys())[zero_idx[0]]) + " vertex has no edges.",
                UserWarning,
            )

        if len(zero_idx) > 1:
            names = []
            for idx in zero_idx:
                names.append(list(self.id_dict.keys())[idx])
            warnings.warn(str(names) + " vertices have no edges.", UserWarning)

    def adjacency_matrix(self, directed=False, weighted=False):
        """Return the adjacency matrix of the graph."""
        # Ensure matrix is symmetric as this is undirected
        if weighted:
            # Symmetrize
            adj = (
                self.adj
                + self.adj.T
                - sp.spdiags(self.adj.diagonal(), 0, m=self.adj.shape)
            )
        else:
            adj = self.adj != 0
            adj = adj + adj.T

        return adj.tocsr()

    def num_edges(self, recompute=False):
        """Compute the number of edges."""
        if not hasattr(self, "_num_edges") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self._num_edges = (adj.sum() + adj.diagonal().sum()) / 2

        return self._num_edges

    def total_weight(self, recompute=False):
        """Compute the sum of all the weights."""
        if not hasattr(self, "_total_weight") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=True)
            self._total_weight = (adj.sum() + adj.diagonal().sum()) / 2

        return self._total_weight

    def degree(self, recompute=False):
        """Compute the undirected degree sequence."""
        if not hasattr(self, "_degree") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self._degree = adj.sum(axis=0)

        return self._degree

    def degree_by_group(self, recompute=False):
        """Compute the undirected degree sequence to and from each group."""
        if not hasattr(self, "_degree_by_group") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)

            id_arr, grp_arr, cnt_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups, dtype=np.int64
            )

            self._degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._degree_by_group

    def strength(self, recompute=False):
        """Compute the total strength sequence."""
        if not hasattr(self, "_strength") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=True)
            self._strength = adj.sum(axis=1)

        return self._strength

    def strength_by_group(self, recompute=False):
        """Compute the total strength sequence to and from each group."""
        if not hasattr(self, "_strength_by_group") or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=True)

            id_arr, grp_arr, val_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups
            )

            self._strength_by_group = sp.coo_array(
                (val_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._strength_by_group

    def average_nn_property(self, prop, selfloops=False, deg_recompute=False):
        """Computes the nearest neighbour average of
        the property array. The array must have the first dimension
        corresponding to the vertex index.
        """
        # Check first dimension of property array is correct
        if prop.shape[0] != self.num_vertices:
            msg = (
                "Property array must have first dimension size be equal to"
                " the number of vertices."
            )
            raise ValueError(msg)

        if selfloops is None:
            selfloops = self.selfloops

        # Compute correct degree
        deg = self.degree(recompute=deg_recompute)

        if sp.issparse(prop):
            prop = prop.todense()

        # It is necessary to select the elements or pickling will fail
        adj = self.adjacency_matrix(directed=False, weighted=False)
        av_nn = self.av_nn_prop(adj.indptr, adj.indices, prop, selfloops)

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        return av_nn

    def average_nn_degree(self, selfloops=False, recompute=False, deg_recompute=False):
        """Computes the average nearest neighbour degree of each node."""
        # Compute property name
        name = "av_nn_degree"
        if not hasattr(self, name) or recompute or deg_recompute:
            if selfloops is None:
                selfloops = self.selfloops

            # Compute correct degree
            deg = self.degree(recompute=deg_recompute)

            res = self.average_nn_property(
                deg, selfloops=selfloops, deg_recompute=False
            )
            setattr(self, name, res)

        return getattr(self, name)

    def to_networkx(self):
        """Return a networkx Graph object for this graph."""
        # Initialize Graph object
        G = nx.Graph(self.adjacency_matrix(directed=False, weighted=True))

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, "group_dict"):
            gr_list = [None] * self.num_groups
            for gr_id, i in self.group_dict.items():
                gr_list[i] = gr_id

            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr_list[gr])

        return G

    @staticmethod
    def get_num_bytes(num_items):
        """Determine the number of bytes needed for storing ids for num_items."""
        return int(max(2 ** np.ceil(np.log2(np.log2(num_items + 1) / 8)), 1))

    @staticmethod
    def generate_id_dict(df, id_col, check_unique=False):
        """Return id dictionary for given dataframe columns."""
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
    @jit(nopython=True)  # pragma: no cover
    def sum_by_group(indptr, indices, values, g_arr, dtype=np.float64):
        """Sums the values of the matrix along the indices axis using the
        provided grouping.
        """
        # Get number of index of first dimension
        N = len(indptr) - 1

        # Convert indices of matrix using group dict
        groups = [g_arr[x] for x in indices]

        # Count group occurrences
        id_arr = []
        grp_arr = []
        sum_arr = []
        for i in range(N):
            m = indptr[i]
            n = indptr[i + 1]
            gcnt = {}
            for g, v in zip(groups[m:n], values[m:n]):
                if g in gcnt:
                    gcnt[g] = gcnt[g] + dtype(v)
                else:
                    gcnt[g] = dtype(v)
            id_arr.extend([i] * len(gcnt.keys()))
            grp_arr.extend(gcnt.keys())
            sum_arr.extend(gcnt.values())
        return id_arr, grp_arr, sum_arr

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def av_nn_prop(indptr, indices, prop, selfloops):
        """Computes the sum of the nearest neighbours' properties."""
        # Get number of index of first dimension
        N = len(indptr) - 1
        res = np.zeros(prop.shape, dtype=np.float64)

        for i in range(N):
            m = indptr[i]
            n = indptr[i + 1]
            for j in indices[m:n]:
                if (i != j) or selfloops:
                    res[i] += prop[j]

        return res


class DiGraph(Graph):
    """General class for directed graphs.

    Note that edges can be weighted or not. If they are not, the strengths
    will be equal to the degrees. The class does not check for the uniqueness
    of the links definitions. If a link is provided multiple times with
    weights, they will be summed.

    Attributes
    ----------
    num_vertices: int
        Number of vertices in the graph.
    adj: numpy.array
        The adjacency matrix of the graph.
    id_dict: dict
        Dictionary to convert original identifiers to new position id.
    id_dtype: numpy.dtype
        Type of the id (e.g. np.uint16).
    num_groups: int  (or None)
        Number of vertex groups.
    group_dict: dict (or none)
        Dictionary to convert v_group columns into numeric ids.
    group_dtype: numpy.dtype
        Type of the group id.
    groups: numpy.array
        Array with the group each node belongs to.
    weighted: bool
        Is true if the edges have an associated weight.

    Methods
    -------
    num_edges:
        Compute the number of edges in the graph.
    total_weight:
        Compute the total sum of the weights of the edges.
    degree:
        Compute the undirected degree sequence.
    out_degree:
        Compute the out degree sequence.
    in_degree:
        Compute the in degree sequence.
    degree_by_group:
        Compute the undirected degree sequence by group as a 2D array.
    out_degree_by_group:
        Compute the out degree sequence by group as a 2D array.
    in_degree_by_group:
        Compute the in degree sequence by group as a 2D array.
    strength:
        Compute the total strength sequence.
    out_strength:
        Compute the out strength sequence.
    in_strength:
        Compute the in strength sequence.
    strength_by_group:
        Compute the total strength sequence by group as a 2D array.
    out_strength_by_group:
        Compute the out strength sequence by group as a 2D array.
    in_strength_by_group:
        Compute the in strength sequence by group as a 2D array.
    average_nn_property:
        Compute the average nearest neighbour property of each node.
    average_nn_degree:
        Compute the average nearest neighbour degree of each node.
    adjacency_matrix:
        Return the adjacency matrix of the graph.
    to_networkx:
        Return a Networkx equivalent.
    """

    def __init__(self, v, e, v_id, src, dst, weight=None, v_group=None):
        """Return a DiGraph object given vertices and edges.

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
        weight: str or None
            identifier column for the weight of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        DiGraph
            the graph object
        """
        super().__init__(
            v, e, v_id=v_id, src=src, dst=dst, weight=weight, v_group=v_group
        )

    def adjacency_matrix(self, directed=True, weighted=False):
        """Return the adjacency matrix of the graph."""
        if directed and weighted:
            adj = self.adj

        elif directed and not weighted:
            adj = self.adj != 0

        elif not directed and weighted:
            # Symmetrize
            adj = (
                self.adj
                + self.adj.T
                - sp.spdiags(self.adj.diagonal(), 0, m=self.adj.shape)
            )

        else:
            adj = self.adj != 0
            adj = adj + adj.T

        return adj.tocsr()

    def num_edges(self, recompute=False):
        """Compute the number of edges."""
        if not hasattr(self, "_num_edges") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=False)
            self._num_edges = adj.sum()

        return self._num_edges

    def total_weight(self, recompute=False):
        """Compute the sum of all the weights."""
        if not hasattr(self, "_total_weight") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=True)
            self._total_weight = adj.sum()

        return self._total_weight

    def out_degree(self, recompute=False):
        """Compute the out degree sequence."""
        if not hasattr(self, "_out_degree") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=False)
            self._out_degree = adj.sum(axis=1)
            self._in_degree = adj.sum(axis=0)

        return self._out_degree

    def in_degree(self, recompute=False):
        """Compute the in degree sequence."""
        if not hasattr(self, "_in_degree") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=False)
            self._out_degree = adj.sum(axis=1)
            self._in_degree = adj.sum(axis=0)

        return self._in_degree

    def out_degree_by_group(self, recompute=False):
        """Compute the out degree sequence to each group."""
        if not hasattr(self, "_out_degree_by_group") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=False)
            id_arr, grp_arr, cnt_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups, dtype=np.int64
            )
            self._out_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()
            adj = adj.tocsc()
            id_arr, grp_arr, cnt_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups, dtype=np.int64
            )
            self._in_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._out_degree_by_group

    def in_degree_by_group(self, recompute=False):
        """Compute the in degree sequence from each group."""
        if not hasattr(self, "_in_degree_by_group") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=False)
            id_arr, grp_arr, cnt_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups, dtype=np.int64
            )
            self._out_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()
            adj = adj.tocsc()
            id_arr, grp_arr, cnt_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups, dtype=np.int64
            )
            self._in_degree_by_group = sp.coo_array(
                (cnt_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._in_degree_by_group

    def out_strength(self, recompute=False):
        """Compute the out strength sequence."""
        if not hasattr(self, "_out_strength") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=True)
            self._out_strength = adj.sum(axis=1)

        return self._out_strength

    def in_strength(self, recompute=False):
        """Compute the in strength sequence."""
        if not hasattr(self, "_in_strength") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=True)
            self._in_strength = adj.sum(axis=0)

        return self._in_strength

    def out_strength_by_group(self, recompute=False):
        """Compute the out strength sequence to each group."""
        if not hasattr(self, "_out_strength_by_group") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=True)
            id_arr, grp_arr, val_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups
            )
            self._out_strength_by_group = sp.coo_array(
                (val_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._out_strength_by_group

    def in_strength_by_group(self, recompute=False):
        """Compute the in strength sequence from each group."""
        if not hasattr(self, "_in_strength_by_group") or recompute:
            adj = self.adjacency_matrix(directed=True, weighted=True).tocsc()
            id_arr, grp_arr, val_arr = self.sum_by_group(
                adj.indptr, adj.indices, adj.data, self.groups
            )
            self._in_strength_by_group = sp.coo_array(
                (val_arr, (id_arr, grp_arr)), shape=(self.num_vertices, self.num_groups)
            ).tocsr()

        return self._in_strength_by_group

    def average_nn_property(
        self, prop, ndir="out", selfloops=False, deg_recompute=False
    ):
        """Computes the nearest neighbour average of the property array.
        The array must have the first dimension corresponding to the vertex
        index. It expects one of three options for the neighbourhood
        direction: 'out', 'in', 'out-in'. The last corresponds to the undirected case.
        """
        # Check first dimension of property array is correct
        if prop.shape[0] != self.num_vertices:
            msg = (
                "Property array must have first dimension size be equal to"
                " the number of vertices."
            )
            raise ValueError(msg)

        if selfloops is None:
            selfloops = self.selfloops

        # Compute correct degree
        if ndir == "out":
            deg = self.out_degree(recompute=deg_recompute)
            adj = self.adjacency_matrix(directed=True, weighted=False)
        elif ndir == "in":
            deg = self.in_degree(recompute=deg_recompute)
            adj = self.adjacency_matrix(directed=True, weighted=False).tocsc()
        elif ndir == "out-in":
            deg = self.degree(recompute=deg_recompute)
            adj = self.adjacency_matrix(directed=False, weighted=False)
        else:
            raise ValueError("Neighbourhood direction not recognised.")

        if sp.issparse(prop):
            prop = prop.todense()

        # It is necessary to select the elements or pickling will fail
        av_nn = self.av_nn_prop(adj.indptr, adj.indices, prop, selfloops)

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        return av_nn

    def average_nn_degree(
        self,
        ndir="out",
        ddir="out",
        selfloops=False,
        recompute=False,
        deg_recompute=False,
    ):
        """Computes the average nearest neighbour degree of each node.
        It expects one of three options for the degree direction and the
        neighbourhood direction: 'out', 'in', 'out-in'. The last corresponds
        to the undirected case.
        """
        # Compute property name
        name = "av_" + ndir.replace("-", "_") + "_nn_d_" + ddir.replace("-", "_")
        if not hasattr(self, name) or recompute or deg_recompute:
            if selfloops is None:
                selfloops = self.selfloops

            # Compute correct degree
            if ddir == "out":
                deg = self.out_degree(recompute=deg_recompute)
            elif ddir == "in":
                deg = self.in_degree(recompute=deg_recompute)
            elif ddir == "out-in":
                deg = self.degree(recompute=deg_recompute)
            else:
                raise ValueError("Degree type not recognised.")

            # It is necessary to select the elements or pickling will fail
            res = self.average_nn_property(
                deg, ndir=ndir, selfloops=selfloops, deg_recompute=False
            )
            setattr(self, name, res)

        return getattr(self, name)

    def to_networkx(self, original=False):
        """Return a networkx DiGraph object for this graph."""
        # Initialize DiGraph object
        G = nx.DiGraph(self.adj)

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, "group_dict"):
            gr_list = [None] * self.num_groups
            for gr_id, i in self.group_dict.items():
                gr_list[i] = gr_id

            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr_list[gr])

        return G


class MultiGraph(Graph):
    """General class for multidimensional graphs with parallel edges.

    This class allows to define multiple edges between the same nodes by adding
    a label to the edges. The labels define a new dimension for adjacency
    matrix which is now a 3D tensor. The adjacency_matrix method will return
    the projection of the tensor on the 2D space, for the full tensor call the
    adjacency_tensor method. A slice of the adjacency tensor along a label is
    called a layer.

    Note that we are adopting here the formalism of the label being on the
    edge and not on the vertex. We assume vertex to have a unique identity.
    This disallows to consider explicitly edges between layers. Each edge is
    therefore defined by the triple (v, u, d) where d is the edge/layer label
    and not by the tuple (v, d, u, e) where d and e are the layers of v and u
    respectively. This case can be constructed implicitly in two ways: either
    by label the edges based on the source and destination layer or by
    defining the vertices multiple time one for each layer they belong to in a
    single dimension graph. In the second case the grouping option allows to
    preserve easily the identity of the node across layers.

    Note that edges can be weighted or not. If they are not, the strengths
    will be equal to the degrees. The class does not check for the uniqueness
    of the links definitions. If a link is provided multiple times with
    weights, they will be summed. This also applies to the case where both the
    (i, j, weight) and symmetric (j, i, weight) are provided.

    Attributes
    ----------
    num_vertices: int
        Number of vertices in the graph.
    num_labels: int
        Number of distinct edge labels.
    adj: numpy.array
        The adjacency tensor of the graph.
    id_dict: dict
        Dictionary to convert original identifiers to new position id.
    id_dtype: numpy.dtype
        Type of the id (e.g. np.uint16).
    label_dtype: numpy.dtype
        The data type of the label internal id.
    num_groups: int  (or None)
        Number of vertex groups.
    group_dict: dict (or none)
        Dictionary to convert v_group columns into numeric ids.
    group_dtype: numpy.dtype
        Type of the group id.
    groups: numpy.array
        Array with the group each node belongs to.
    weighted: bool
        Is true if the edges have an associated weight.

    Methods
    -------
    num_edges:
        Compute the number of edges in the graph.
    num_edges_label:
        Compute the number of edges by label (in order)
    total_weight:
        Compute the total sum of the weights of the edges.
    total_weight_label:
        Compute the total sum of the weights by label.
    degree:
        Compute the undirected degree sequence.
    degree_by_label:
        Compute the degree of each vertex by label.
    degree_by_group:
        Compute the undirected degree sequence by group as a 2D array.
    strength:
        Compute the total strength sequence.
    strength_by_label:
        Compute the strength of each vertex by label.
    strength_by_group:
        Compute the total strength sequence by group as a 2D array.
    average_nn_property:
        Compute the average nearest neighbour property of each node.
    average_nn_degree:
        Compute the average nearest neighbour degree of each node.
    adjacency_matrix:
        Return the adjacency matrix of the graph.
    adjacency_tensor:
        Return the adjacency tensor of the graph.
    to_networkx:
        Return a Networkx equivalent.
    """

    def __init__(self, v, e, v_id, src, dst, edge_label, weight=None, v_group=None):
        """Return a MultiGraph object given vertices and edges.

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
        weight: str or None
            identifier column for the weight of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        MultiGraph
            the graph object
        """
        super().__init__(
            v, e, v_id=v_id, src=src, dst=dst, weight=weight, v_group=v_group
        )

        # If column names are passed as lists with one elements extract str
        if isinstance(edge_label, list) and len(edge_label) == 1:
            edge_label = edge_label[0]

        # Get dictionary of label to numeric internal label
        if isinstance(edge_label, list):
            e["_label"] = list(zip(*[e[x] for x in edge_label]))
        else:
            e["_label"] = e[edge_label]

        self.label_dict = self.generate_id_dict(e, "_label")
        self.num_labels = len(self.label_dict.keys())
        num_bytes = self.get_num_bytes(self.num_labels)
        self.label_dtype = np.dtype("u" + str(num_bytes))

        # Convert labels to construct adjacency matrix by layer
        e["_label"] = e["_label"].map(lambda x: self.label_dict.get(x))
        lbl_array = e["_label"].values.astype(self.label_dtype)
        src_array = e["_src"].values.astype(self.id_dtype)
        dst_array = e["_dst"].values.astype(self.id_dtype)

        # Construct adjacency tensor with dimensions (label, src, dst)
        if weight is not None:
            val_array = e[weight].values

            # Define adjacency tensor as list of sparse matrices
            self.adj = []
            for lbl in range(self.num_labels):
                ind = lbl_array == lbl
                self.adj.append(
                    sp.coo_array(
                        (val_array[ind], (src_array[ind], dst_array[ind])),
                        shape=(self.num_vertices, self.num_vertices),
                    ).tocsr()
                )

        else:
            # Define adjacency tensor as list of sparse matrices
            self.adj = []
            for lbl in range(self.num_labels):
                ind = lbl_array == lbl
                self.adj.append(
                    sp.coo_array(
                        (
                            np.ones(len(src_array[ind]), bool),
                            (src_array[ind], dst_array[ind]),
                        ),
                        shape=(self.num_vertices, self.num_vertices),
                    ).tocsr()
                )

    def adjacency_matrix(self, directed=False, weighted=False):
        """Return the adjacency matrix of the graph."""
        # Project tensor to two dimensions
        adj = self.adj[0]
        for lbl in range(1, self.num_labels):
            adj += self.adj[lbl]

        # Ensure matrix is symmetric as this is undirected
        if weighted:
            # Symmetrize
            adj = adj + adj.T - sp.spdiags(adj.diagonal(), 0, m=adj.shape)

        else:
            adj = adj != 0
            adj = adj + adj.T

        return adj

    def adjacency_tensor(self, directed=False, weighted=False):
        """Return the adjacency tensor."""
        # Ensure matrix for each layer is symmetric as this is undirected
        if weighted:
            # Symmetrize
            sym = []
            for lbl in range(self.num_labels):
                adj = self.adj[lbl]
                sym.append(adj + adj.T - sp.spdiags(adj.diagonal(), 0, m=adj.shape))

        else:
            sym = []
            for lbl in range(self.num_labels):
                adj = self.adj[lbl] != 0
                sym.append(adj + adj.T)

        return sym

    def num_edges_label(self, recompute=False):
        """Compute the number of edges."""
        if not hasattr(self, "_num_edges_label") or recompute:
            adj = self.adjacency_tensor(directed=False, weighted=False)
            ne_lbl = []
            for lbl in range(self.num_labels):
                ne_lbl.append((adj[lbl].nnz + adj[lbl].diagonal().sum()) / 2)

            self._num_edges_label = np.array(ne_lbl)

        return self._num_edges_label

    def total_weight_label(self, recompute=False):
        """Compute the sum of all the weights."""
        if not hasattr(self, "_total_weight_label") or recompute:
            adj = self.adjacency_tensor(directed=False, weighted=True)
            w_lbl = []
            for lbl in range(self.num_labels):
                w_lbl.append((adj[lbl].sum() + adj[lbl].diagonal().sum()) / 2)

            self._total_weight_label = np.array(w_lbl)

        return self._total_weight_label

    def degree_by_label(self, recompute=False):
        """Compute the degree sequence by label."""
        if not hasattr(self, "_degree_by_label") or recompute:
            adj = self.adjacency_tensor(directed=False, weighted=False)
            d_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                d_label[lbl, :] = adj[lbl].sum(axis=1)

            self._degree_by_label = d_label.T.tocsr()

        return self._degree_by_label

    def strength_by_label(self, recompute=False):
        """Compute the strength sequence by label."""
        if not hasattr(self, "_strength_by_label") or recompute:
            adj = self.adjacency_tensor(directed=False, weighted=True)
            s_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                s_label[lbl, :] = adj[lbl].sum(axis=1)

            self._strength_by_label = s_label.T.tocsr()

        return self._strength_by_label

    def to_networkx(self):
        """Return a networkx MultiGraph equivalent object for this graph."""
        # Initialize object
        G = nx.MultiGraph()

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, "group_dict"):
            gr_list = [None] * self.num_groups
            for gr_id, i in self.group_dict.items():
                gr_list[i] = gr_id

            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr_list[gr])

        # Create label list to assign original values
        lbl_list = [None] * self.num_labels
        for lbl_id, i in self.label_dict.items():
            lbl_list[i] = lbl_id

        # Add edges per layer
        for k in range(self.num_labels):
            adj = self.adj[k].tocoo()
            lbl = np.empty(adj.nnz)
            lbl[:] = k
            G.add_edges_from(
                zip(
                    adj.row,
                    adj.col,
                    lbl,
                    [dict(weight=i, label=lbl_list[k]) for i in adj.data],
                )
            )

        return G


class MultiDiGraph(MultiGraph, DiGraph):
    """General class for multidimensional directed graphs with parallel edges.

    This class allows to define multiple edges between the same nodes by adding
    a label to the edges. The labels define a new dimension for adjacency
    matrix which is now a 3D tensor. The adjacency_matrix method will return
    the projection of the tensor on the 2D space, for the full tensor call the
    adjacency_tensor method. A slice of the adjacency tensor along a label is
    called a layer.

    Note that we are adopting here the formalism of the label being on the
    edge and not on the vertex. We assume vertex to have a unique identity.
    This disallows to consider explicitly edges between layers. Each edge is
    therefore defined by the triple (v, u, d) where d is the edge/layer label
    and not by the tuple (v, d, u, e) where d and e are the layers of v and u
    respectively. This case can be constructed implicitly in two ways: either
    by label the edges based on the source and destination layer or by
    defining the vertices multiple time one for each layer they belong to in a
    single dimension graph. In the second case the grouping option allows to
    preserve easily the identity of the node across layers.

    Note that edges can be weighted or not. If they are not, the strengths
    will be equal to the degrees. The class does not check for the uniqueness
    of the links definitions. If a link is provided multiple times with
    weights, they will be summed.


    Attributes
    ----------
    num_vertices: int
        Number of vertices in the graph.
    num_labels: int
        Number of distinct edge labels.
    adj: numpy.array
        The adjacency tensor of the graph.
    id_dict: dict
        Dictionary to convert original identifiers to new position id.
    id_dtype: numpy.dtype
        Type of the id (e.g. np.uint16).
    label_dtype: numpy.dtype
        The data type of the label internal id.
    num_groups: int  (or None)
        Number of vertex groups.
    group_dict: dict (or none)
        Dictionary to convert v_group columns into numeric ids.
    group_dtype: numpy.dtype
        Type of the group id.
    groups: numpy.array
        Array with the group each node belongs to.
    weighted: bool
        Is true if the edges have an associated weight.

    Methods
    -------
    num_edges:
        Compute the number of edges in the graph.
    num_edges_label:
        Compute the number of edges by label (in order)
    total_weight:
        Compute the total sum of the weights of the edges.
    total_weight_label:
        Compute the total sum of the weights by label.
    degree:
        Compute the undirected degree sequence.
    out_degree:
        Compute the out degree sequence.
    in_degree:
        Compute the in degree sequence.
    degree_by_label:
        Compute the degree of each vertex by label.
    out_degree_by_label:
        Compute the out degree of each vertex by label.
    in_degree_by_label:
        Compute the in degree of each vertex by label.
    degree_by_group:
        Compute the undirected degree sequence by group as a 2D array.
    out_degree_by_group:
        Compute the out degree sequence by group as a 2D array.
    in_degree_by_group:
        Compute the in degree sequence by group as a 2D array.
    strength:
        Compute the total strength sequence.
    out_strength:
        Compute the out strength sequence.
    in_strength:
        Compute the in strength sequence.
    strength_by_label:
        Compute the strength of each vertex by label.
    out_strength_by_label:
        Compute the out strength of each vertex by label.
    in_strength_by_label:
        Compute the in strength of each vertex by label.
    strength_by_group:
        Compute the total strength sequence by group as a 2D array.
    out_strength_by_group:
        Compute the out strength sequence by group as a 2D array.
    in_strength_by_group:
        Compute the in strength sequence by group as a 2D array.
    average_nn_property:
        Compute the average nearest neighbour property of each node.
    average_nn_degree:
        Compute the average nearest neighbour degree of each node.
    adjacency_matrix:
        Return the adjacency matrix of the graph.
    adjacency_tensor:
        Return the adjacency tensor of the graph.
    to_networkx:
        Return a Networkx equivalent.
    """

    def __init__(self, v, e, v_id, src, dst, edge_label, weight=None, v_group=None):
        """Return a MultiGraph object given vertices and edges.

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
        weight: str or None
            identifier column for the weight of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        MultiDiGraph
            the graph object
        """
        super().__init__(
            v,
            e,
            v_id=v_id,
            src=src,
            dst=dst,
            edge_label=edge_label,
            weight=weight,
            v_group=v_group,
        )

    def adjacency_matrix(self, directed=True, weighted=False):
        """Return the adjacency matrix of the graph."""
        # Project tensor to two dimensions
        adj = self.adj[0]
        for lbl in range(1, self.num_labels):
            adj += self.adj[lbl]

        if directed and not weighted:
            adj = adj != 0

        elif not directed and weighted:
            # Symmetrize
            adj = adj + adj.T - sp.spdiags(adj.diagonal(), 0, m=adj.shape)

        elif not directed and not weighted:
            adj = adj != 0
            adj = adj + adj.T

        return adj

    def adjacency_tensor(self, directed=True, weighted=False):
        """Return the adjacency tensor."""
        if directed and weighted:
            adj = self.adj

        elif directed and not weighted:
            adj = []
            for lbl in range(self.num_labels):
                adj.append(self.adj[lbl] != 0)

        elif not directed and weighted:
            # Symmetrize
            adj = []
            for lbl in range(self.num_labels):
                tmp = self.adj[lbl]
                adj.append(tmp + tmp.T - sp.spdiags(tmp.diagonal(), 0, m=tmp.shape))

        else:
            adj = []
            for lbl in range(self.num_labels):
                tmp = self.adj[lbl] != 0
                adj.append(tmp + tmp.T)

        return adj

    def num_edges_label(self, recompute=False):
        """Compute the number of edges."""
        if not hasattr(self, "_num_edges_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=False)
            self._num_edges_label = np.array([a.nnz for a in adj])

        return self._num_edges_label

    def total_weight_label(self, recompute=False):
        """Compute the sum of all the weights."""
        if not hasattr(self, "_total_weight_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=True)
            self._total_weight_label = np.array([a.sum() for a in adj])

        return self._total_weight_label

    def out_degree_by_label(self, recompute=False):
        """Compute the out degree sequence by label."""
        if not hasattr(self, "_out_degree_by_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=False)
            d_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                d_label[lbl, :] = adj[lbl].sum(axis=1)

            self._out_degree_by_label = d_label.T.tocsr()

        return self._out_degree_by_label

    def in_degree_by_label(self, recompute=False):
        """Compute the in degree sequence by label."""
        if not hasattr(self, "_in_degree_by_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=False)
            d_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                d_label[lbl, :] = adj[lbl].sum(axis=0)

            self._in_degree_by_label = d_label.T.tocsr()

        return self._in_degree_by_label

    def out_strength_by_label(self, recompute=False):
        """Compute the out strength sequence by label."""
        if not hasattr(self, "_out_strength_by_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=True)
            s_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                s_label[lbl, :] = adj[lbl].sum(axis=1)

            self._out_strength_by_label = s_label.T.tocsr()

        return self._out_strength_by_label

    def in_strength_by_label(self, recompute=False):
        """Compute the in strength sequence by label."""
        if not hasattr(self, "_in_strength_by_label") or recompute:
            adj = self.adjacency_tensor(directed=True, weighted=True)
            s_label = sp.lil_array((self.num_labels, self.num_vertices))
            for lbl in range(self.num_labels):
                s_label[lbl, :] = adj[lbl].sum(axis=0)

            self._in_strength_by_label = s_label.T.tocsr()

        return self._in_strength_by_label

    def to_networkx(self):
        """Return a networkx MultiDiGraph equivalent object for this graph."""
        # Initialize object
        G = nx.MultiDiGraph()

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, "group_dict"):
            gr_list = [None] * self.num_groups
            for gr_id, i in self.group_dict.items():
                gr_list[i] = gr_id

            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr_list[gr])

        # Create label list to assign original values
        lbl_list = [None] * self.num_labels
        for lbl_id, i in self.label_dict.items():
            lbl_list[i] = lbl_id

        # Add edges per layer
        for k in range(self.num_labels):
            adj = self.adj[k].tocoo()
            lbl = np.empty(adj.nnz)
            lbl[:] = k
            G.add_edges_from(
                zip(
                    adj.row,
                    adj.col,
                    lbl,
                    [dict(weight=i, label=lbl_list[k]) for i in adj.data],
                )
            )

        return G
