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


@jit(forceobj=True, parallel=True)
def vector_fitness_link_prob_stripe_one_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0

    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i, 0]
        sect_out = out_strength[i, 1]
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j, 0]
            sect_in = in_strength[j, 1]
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


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


@jit(forceobj=True, parallel=True)
def vector_fitness_link_prob_stripe_mult_z(out_strength, in_strength, z):
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
                                          z, dict_sectors):
    """
    Function computing the Probability Matrix of the Cimi block model
    with just one parameter controlling for the density.
    """

    N = len(np.unique(out_strength[:, 0]))
    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i, 0]
        sect_out = out_strength[i, 1]
        s_out = out_strength[i, 2]
        sect_node1 = dict_sectors[ind_out]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j, 0]
            sect_in = in_strength[j, 1]
            s_in = in_strength[j, 2]
            sect_node2 = dict_sectors[ind_in]
            if ((ind_out != ind_in) & (sect_out == sect_node2) &
               (sect_in == sect_node1)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


@jit(forceobj=True, parallel=True)
def vector_fitness_link_prob_block_one_z(out_strength, in_strength,
                                         z, dict_sectors):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0

    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i, 0]
        sect_out = out_strength[i, 1]
        s_out = out_strength[i, 2]
        sect_node1 = dict_sectors[ind_out]
        for j in prange(in_strength.shape[0]):
            ind_in = in_strength[j, 0]
            sect_in = in_strength[j, 1]
            s_in = in_strength[j, 2]
            sect_node2 = dict_sectors[ind_in]
            if ((ind_out != ind_in) & (sect_out == sect_node2) &
               (sect_in == sect_node1)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_block_multiple_z(out_strength, in_strength,
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


@jit(forceobj=True, parallel=True)
def vector_fitness_link_prob_block_multiple_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    n_sector = int(np.sqrt(max(z.shape)))
    p = np.zeros(shape=(n_sector, n_sector), dtype=np.float)
    z = np.reshape(z, newshape=(n_sector, n_sector))
    for i in np.arange(out_strength.shape[0]):
        ind_out = out_strength[i, 0]
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
