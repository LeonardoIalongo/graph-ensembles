""" This module defines the functions that allow for the construction of 
network ensembles from partial information. They can be used for 
reconstruction, filtering or pattern detection among others. """

import numpy as np 
import pandas as pd 
import scipy.sparse as sp

def get_strenghts(edges, vertices, group_col=None, group_dir='in'):
    """ Returns the strength sequence for the given network. If a group_col
    is given then it returns a vector for each strength where each element is 
    the strength related to each group. You can specify whether the grouping
    applies only to the 'in', 'out', or 'all' edges through group_dir."""

    if group_col is None:
        # If no group is specified return total strength
        out_strenght = edges.groupby(['src'], as_index=False).agg(
            {'weight': sum}).rename(
            columns={"weight": "out_strength", "src": "id"})

        in_strength = edges.groupby(['dst'], as_index=False).agg(
        {'weight': sum}).rename(
        columns={"weight": "in_strength", "dst": "id"})

        strength = out_strenght.join(
            in_strength.set_index('id'), on='id', how='outer').fillna(0)
    else:
        if group_dir in ['out', 'all']:
            # Get group of dst edge
            temp = edges.join(vertices.set_index('id'), on='dst', how='left')
            out_strenght = temp.groupby(['src', group_col],
                as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "out_strength", "src": "id"})
        else:
            out_strenght = edges.groupby(['src'], as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "out_strength", "src": "id"})

        if group_dir in ['in', 'all']:
            # Get group of src edge
            temp = edges.join(vertices.set_index('id'), on='src', how='left')
            in_strength = temp.groupby(['dst', group_col], 
                as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "in_strength", "dst": "id"})
        else:
            in_strength = edges.groupby(['dst'], as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "in_strength", "dst": "id"})

        strength = out_strenght.join(
            in_strength.set_index('id'), on='id', how='outer')

    
    return strength


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


