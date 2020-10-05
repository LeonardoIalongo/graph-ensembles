import numpy as np
from numba import jit


@jit(nopython=True,parallel=True)
def vector_fitness_prob_array_stripe_one_z(out_strength, in_strength, z):
    """
    Function computing the Probability Matrix of the Cimi stripe model 
    with just one parameter controlling for the density.
    """

    p = np.zeros((N, N), dtype=np.float64)
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i,0]
        sect_out = out_strength[i,1]
        s_out = out_strength[i,2]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j,0]
            sect_in = in_strength[j,1]
            s_in = in_strength[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)

    return p


@jit(forceobj=True,parallel=True)
def vector_fitness_link_prob_stripe_one_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i,0]
        sect_out = out_strength[i,1]
        s_out = out_strength[i,2]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j,0]
            sect_in = in_strength[j,1]
            s_in = in_strength[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(forceobj=True)
def assign_weights_cimi_stripe_one_z(p, out_strength, in_strength, expected=True):
    """Function returning the weighted adjacency matrix of the Cimi stripe model
    with just one global parameter z controlling for the density. Depending on the value of 
    "expected" the weighted adjacency matrix can be the expceted one or just an ensemble realisation.

    Parameters
    ----------
    p: scipy.sparse.matrix or np.ndarray or list of lists
        the binary probability matrix
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    group_dict: dict
        a dictionary that given the index of a node returns its group
    expected: bool
        If True the strength of each link is the expected one
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
    W = np.ones((N, N), dtype=np.float64)
    
    if out_strength.ndim != in_strength.ndim:
        raise ValueError('Number of sectors has to be the same for strength-in and out.')
    
    else:
        if expected:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, :]
                        tmp_sec = np.nonzero(s_i)[0][0]
                        s_j = in_strength[j, :]
                        tmp = np.dot(s_i,s_j)
                        if tmp ==0:
                            continue
                        tot_w = np.sum(out_strength[:,tmp_sec])
                        W[i, j] = (tmp)/(tot_w)
            
        else:    
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        if p[i,j] > np.random.random():
                            s_i = out_strength[i, :]
                            tmp_sec = np.nonzero(s_i)[0][0]
                            s_j = in_strength[j, :]
                            tmp = np.dot(s_i,s_j)
                            if tmp ==0:
                                continue
                            tot_w = np.sum(out_strength[:,tmp_sec])
                            W[i, j] = (tmp)/(tot_w*p[i,j])

    return W