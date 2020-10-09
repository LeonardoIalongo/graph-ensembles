import numpy as np
from numba import jit
from numba import prange


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_stripe_one_z(out_strength, in_strength, z, N):
    """
    Function computing the Probability Matrix of the Cimi stripe model
    with just one parameter controlling for the density.
    """
    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)

    return p


#@jit(forceobj=True, parallel=False)
def expected_links_stripe_one_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(nopython=True)
def assign_weights_cimi_stripe_one_z(p, out_strength, in_strength, N, group_array, expected=True):
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

    W = np.zeros((N, N), dtype=np.float64)
    strengths_stripe = weights_for_stripe( out_strength, group_array)
    if expected:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_out = int(out_strength[i, 1])
            s_out = out_strength[i, 2]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_in = int(in_strength[j, 1])
                s_in = in_strength[j, 2]
                if (ind_out != ind_in) & (sect_out == sect_in):
                    tot_w = strengths_stripe[sect_out]
                    tmp = s_out*s_in
                    W[ind_out, ind_in] = (tmp)/(tot_w * p[ind_out,ind_in])
    else:    
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_out = int(out_strength[i, 1])
            s_out = out_strength[i, 2]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_in = int(in_strength[j, 1])
                s_in = in_strength[j, 2]
                if (ind_out != ind_in) & (sect_out == sect_in):
                    if p[ind_out,ind_in] > np.random.random():
                        tot_w = strengths_stripe[sect_out]
                        tmp = s_out*s_in
                        W[ind_out, ind_in] = (tmp)/(tot_w)
    return W


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_stripe_mult_z(out_strength, in_strength, z, N):
    """
    Function computing the Probability Matrix of the Cimi stripe model
    with just multiple parameters controlling for the density: one for
    each sector/stripe.
    """
    p = np.zeros((N, N), dtype=np.float64)
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = z[sect_out]*s_out*s_in
                p[ind_out, ind_in] = tmp / (1 + tmp)
    return p


#@jit(forceobj=True, parallel=True)
def expected_links_stripe_mult_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the N parameters z controlling for the density
    of each stripe.
    """

    p = np.zeros(len(z))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = z[sect_out]*s_out*s_in
                p[sect_out] += tmp / (1 + tmp)
    return p


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_block_one_z(out_strength, in_strength,
                                          z, N):
    """
    Function computing the Probability Matrix of the Cimi block model
    with just one parameter controlling for the density.
    """

    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_node_i = int(out_strength[i, 1])
        sect_out = int(out_strength[i, 2])
        s_out = out_strength[i, 3]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_node_j = int(in_strength[j, 1])
            sect_in = int(in_strength[j, 2])
            s_in = in_strength[j, 3]
            if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


#@jit(forceobj=True, parallel=True)
def expected_links_block_one_z(out_strength, in_strength,
                                         z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_node_i = int(out_strength[i, 1])
        sect_out = int(out_strength[i, 2])
        s_out = out_strength[i, 3]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_node_j = int(in_strength[j, 1])
            sect_in = int(in_strength[j, 2])
            s_in = in_strength[j, 3]
            if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(nopython=True)
def assign_weights_cimi_block_one_z(p, out_strength, in_strength, N, strengths_block, expected=True):
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
    strengths_block: np.ndarray
        total strengths between every couple of groups
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
    
    W = np.zeros((N, N), dtype=np.float64)
    strengths_block = weights_for_block(out_strength, group_array)
    if expected:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_node_i = int(out_strength[i, 1])
            sect_out = int(out_strength[i, 2])
            s_out = out_strength[i, 3]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_node_j = int(in_strength[j, 1])
                sect_in = int(in_strength[j, 2])
                s_in = in_strength[j, 3]
                if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                    tot_w = strengths_block[sect_node_i,sect_node_j]
                    tmp = s_out*s_in
                    W[ind_out, ind_in] = (tmp)/(tot_w * p[ind_out,ind_in])
    else:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_node_i = int(out_strength[i, 1])
            sect_out = int(out_strength[i, 2])
            s_out = out_strength[i, 3]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_node_j = int(in_strength[j, 1])
                sect_in = int(in_strength[j, 2])
                s_in = in_strength[j, 3]
                if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                    if p[ind_out, ind_in] > np.random.random():
                        tot_w = strengths_block[sect_out, sect_in]
                        tmp = s_out*s_in
                        W[ind_out, ind_in] = (tmp)/(tot_w * p[ind_out, ind_in])

    return W


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_block_mult_z(out_strength, in_strength,
                                               z, N):
    """
    Function computing the Probability Matrix of the Cimi block model
    with just one parameter controlling for the density.
    """
    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


#@jit(forceobj=True, parallel=True)
def expected_links_block_mult_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    n_sector = int(np.sqrt(max(z.shape)))
    p = np.zeros(shape=(n_sector, n_sector), dtype=np.float)
    z = np.reshape(z, newshape=(n_sector, n_sector))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[sect_out, sect_in] += tmp2 / (1 + tmp2)
    return np.reshape(p, -1)