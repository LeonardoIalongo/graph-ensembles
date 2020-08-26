""" This module defines the functions that allow for the construction of 
network ensembles from partial information. They can be used for 
reconstruction, filtering or pattern detection among others. """

import warnings
import numpy as np 
import pandas as pd 
import scipy.sparse as sp

def get_strenghts(edges, vertices, group_col=None, group_dir='in'):
    """Return the in and out strength sequences for the given network 
    specified by an edge and vertex list as pandas dataframes. 

    If a group_col is given then it returns a vector for each strength where
    each element is the strength related to each group. You can specify 
    whether the grouping applies only to the 'in', 'out', or 'all' edges through group_dir. It also returns a dictionary that returns the group 
    index give the node index and a another that given the identifier of the 
    node, returns the index of it.
    """

    # Check that there are no duplicates in vertex definitions
    if any(vertices.loc[:, 'id'].duplicated()):
        raise ValueError('Duplicated node definitions.')

    # Check no duplicate edges
    if any(edges.loc[:, ['src', 'dst']].duplicated()):
        raise ValueError('There are duplicated edges.')

    # Construct dictionaries
    if group_col is None:
        i = 0
        index_dict = {}
        for index, row in vertices.iterrows():
            index_dict[row.id] = i
            i += 1
        N = len(vertices)
    
        out_temp = edges.groupby(['src']).agg({'weight': sum})
        out_strength = np.zeros(N)
        for index, row in out_temp.iterrows():
            out_strength[index_dict[index]] = row.weight 

        in_temp = edges.groupby(['dst']).agg({'weight': sum})
        in_strength = np.zeros(N)
        for index, row in in_temp.iterrows():
            in_strength[index_dict[index]] = row.weight 

        return out_strength, in_strength, index_dict

    else:
        i = 0
        j = 0
        index_dict = {}
        group_dict = {} 
        group_list = vertices.loc[:, group_col].unique().tolist()
        for index, row in vertices.iterrows():
            index_dict[row.id] = i
            group_dict[i] = group_list.index(row[group_col])
            i += 1
            j += 1
        N = len(vertices)
        G = len(group_list)

        if group_dir in ['out', 'all']:
            out_strength = np.zeros((N, G))
            for index, row in edges.iterrows():
                i = index_dict[row.src]
                j = group_dict[index_dict[row.src]]
                out_strength[i, j] += row.weight
        else:
            out_temp = edges.groupby(['src']).agg({'weight': sum})
            out_strength = np.zeros(N)
            for index, row in out_temp.iterrows():
                out_strength[index_dict[index]] = row.weight 

        if group_dir in ['in', 'all']:
            in_strength = np.zeros((N, G))
            for index, row in edges.iterrows():
                i = index_dict[row.dst]
                j = group_dict[index_dict[row.src]]
                in_strength[i,j] += row.weight
        else:
            in_temp = edges.groupby(['dst']).agg({'weight': sum})
            in_strength = np.zeros(N)
            for index, row in in_temp.iterrows():
                in_strength[index_dict[index]] = row.weight 

        return out_strength, in_strength, index_dict, group_dict


def fitness_link_prob(out_strength, in_strength, z, N, group_dict=None):
    """Compute the link probability matrix given the in and out strength 
    sequence, the density parameter z, and the number of vertices N. 

    The out and in strength sequences should be numpy arrays of 1 dimension. 
    If a group dictionary is specified then it will be assumed that the 
    array will now be 2-dimensional and that the row relates to the node index
    while the column refers to the group. If there is only one dimension it is
    assumed that it is the total strength.
    
    Parameters
    ----------
    out_strength : np.ndarray
        the out strength sequence of graph
    in_strength : np.ndarray
        the in strength sequence of graph
    z: np.float64
        the density parameter of the fitness model
    N: np.int64
        the number of vertices in the graph
    group_dict: dict
        a dictionary that given the index of a node returns its group

    Returns
    -------
    scipy.sparse.lil_matrix
        the link probability matrix

    TODO: Currently implemented with numpy arrays and standard iteration over 
    all indices. Consider allowing for sparse matrices in case of groups and to avoid iteration over all indices.
    """

    # Initialize empty result 
    p = sp.lil_matrix((N, N), dtype=np.float64)

    if group_dict is None:
        if (out_strength.ndim > 1) or (in_strength.ndim > 1):
            raise ValueError('A group dict was not provided but the strength '
                + 'sequence is a vector.')   
        else:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i]
                        s_j = in_strength[j]
                        p[i,j] = z*s_i*s_j / (1 + z*s_i*s_j)
    else:
        if (out_strength.ndim > 1) and (in_strength.ndim > 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, group_dict[j]]
                        s_j = in_strength[j, group_dict[i]]
                        p[i,j] = z*s_i*s_j / (1 + z*s_i*s_j)
        elif (out_strength.ndim > 1) and (in_strength.ndim == 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, group_dict[j]]
                        s_j = in_strength[j]
                        p[i,j] = z*s_i*s_j / (1 + z*s_i*s_j)
        elif (out_strength.ndim == 1) and (in_strength.ndim > 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i]
                        s_j = in_strength[j, group_dict[i]]
                        p[i,j] = z*s_i*s_j / (1 + z*s_i*s_j)
        else:
            raise ValueError('A group dict was provided but no vector' + 
                ' strength sequence is available.')        

    return p


def expected_number_edges(strength, z, num_vertices):
    result = 0
    for i in np.arange(num_vertices):
        for j in np.arange(num_vertices):
            if i != j:
                result += fitness_link_prob(strength(i), strength(j), z)

    return result


