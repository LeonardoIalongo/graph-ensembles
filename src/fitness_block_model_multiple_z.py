import numpy as np
from numba import jit


@jit(nopython=True,parallel=True)
def vector_fitness_prob_array_stripe_mult_z(out_strength, in_strength, z):
    """
    Function computing the Probability Matrix of the Cimi stripe model 
    with just multiple parameters controlling for the density: one for 
    each sector/stripe.
    """
    N = len(np.unique(out_strength[:,0]))
    p = np.zeros((N, N), dtype=np.float64)
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            if (ind_out != ind_in) & (sect_out==sect_in):
                tmp = z[sect_out]*s_out*s_in
                p[ind_out, ind_in] =  tmp / (1 + tmp)
    return p


@jit(forceobj=True,parallel=True)
def vector_fitness_link_prob_stripe_mult_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the N parameters z controlling for the density
    of each stripe.
    """
    
    p = np.zeros(len(z))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            if (ind_out != ind_in) & (sect_out==sect_in):
                tmp = z[sect_out]*s_out*s_in
                p[sect_out] += tmp / (1 + tmp)
    return p


def assign_weights_cimi_block_mult_z(p, out_strength, in_strength, expected=True):
    """Function returning the weighted adjacency matrix of the Cimi block model
    with multiple parameters controlling for the density between blocks. Depending on the value of 
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
        If True the strength of each link is the expected one otherwise
        it is just a single realisation
    Returns
    -------
    np.ndarrays
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
    
    if out_strength.shape != in_strength.shape:
        raise ValueError('Number of sectors has to be the same for strength-in and out.')
    
    else:
        if expected:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i]
                        s_j = in_strength[j]
                        tot_w = out_strength.sum()
                        W[i, j] = (s_i*s_j)/(tot_w)
        else:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        if p[i,j] > np.random.random():
                            s_i = out_strength[i]
                            s_j = in_strength[j]
                            tot_w = out_strength.sum()
                            W[i, j] = (s_i*s_j)/(tot_w*p[i,j])

    return W