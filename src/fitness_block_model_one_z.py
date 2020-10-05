import numpy as np
from numba import jit


@jit(nopython=True,parallel=True)
def vector_fitness_prob_array_block_one_z(out_strength, in_strength, z):
    """
    Function computing the Probability Matrix of the Cimi block model 
    with just one parameter controlling for the density.
    """
    
    N = len(np.unique(out_strength[:,0]))
    p = np.zeros((N, N), dtype=np.float64)
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i,0]
        sect_node1 = out_strength[i,1]
        sect_out = out_strength[i,2]
        s_out = out_strength[i,3]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j,0]
            sect_node2 = in_strength[j,1]
            sect_in = in_strength[j,2]
            s_in = in_strength[j,3]
            if (ind_out != ind_in)&(sect_out==sect_node2)&(sect_in==sect_node1):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp / (1 + tmp)
    return p


@jit(forceobj=True,parallel=True)
def vector_fitness_link_prob_block_one_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i,0]
        sect_node1 = out_strength[i,1]
        sect_out = out_strength[i,2]
        s_out = out_strength[i,3]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j,0]
            sect_node2 = in_strength[j,1]
            sect_in = in_strength[j,2]
            s_in = in_strength[j,3]
            if (ind_out != ind_in)&(sect_out==sect_node2)&(sect_in==sect_node1):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(forceobj=True)
def assign_weights_cimi_block_one_z(p, out_strength, in_strength, group_dict, expected=True):
    """Function returning the weighted adjacency matrix of the Cimi block model
    with just one global parameter z controlling for the density. Depending on the value of 
    "expected" the weighted adjacency matrix can be the expceted one or just an ensemble realisation.

    Parameters
    ----------
    p: scipy.sparse.matrix or np.ndarray or list of lists
        the binary probability matrix
    out_strength: np.ndarray
        the out strength sequence of graph organised by sector
    in_strength: np.ndarray
        the in strength sequence of graph organised by sector
    group_dict: dict
        a dictionary that given the index of a node returns its group
    expected: bool
        If True the strength of each link is the expected one otherwise
        it is just a single realisation
    Returns
    -------
    np.ndarray
        Depending on the value of expected, returns the expected weighted
        matrix or an ensemble realisation

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider allowing for sparse matrices in case of groups and
    to avoid iteration over all indices.
    """
    
    N = max(out_strength.shape)
    if N != max(in_strength.shape):
        raise ValueError('Number of nodes according to data provided does not'
                         ' match.')

    # Initialize empty result
    W = np.zeros((N, N), dtype=np.float64)
    
    if out_strength.ndim != in_strength.ndim:
        raise ValueError('Number of sectors has to be the same for strength-in and out.')
    
    else:
        nodes_group = {}
        for i,v in group_dict.items():
            nodes_group[v] = [i] if v not in nodes_group.keys() else nodes_group[v] + [i]
        
        group_array = dict_sect_to_array(group_dict)

        if expected:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, group_array[j]]
                        s_j = in_strength[j, group_array[i]]
                        tot_w = np.sum(out_strength[nodes_group[group_array[i]],group_array[j]])
                        W[i, j] = (s_i*s_j)/(tot_w)
        else:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        if p[i,j] > np.random.random():
                            s_i = out_strength[i, group_array[j]]
                            s_j = in_strength[j, group_array[i]]
                            tot_w = np.sum(out_strength[nodes_group[group_array[i]],group_array[j]])
                            W[i, j] = (s_i*s_j)/(tot_w*p[i,j])

    return W