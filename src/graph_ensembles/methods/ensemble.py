import numpy as np
from random import random
from numba import jit, prange
from math import ceil, exp, isinf, sqrt


# --------------- RANDOM GRAPH METHODS ---------------
@jit(nopython=True)
def random_graph(n, p):
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """

    if n > 10:
        max_len = int(ceil(n*(n-1)*p + 3*sqrt(n*(n-1)*p*(1-p))))
        a = np.empty((max_len, 2), dtype=np.uint64)
    else:
        a = np.empty((n*(n-1), 2), dtype=np.uint64)
    count = 0
    for i in range(n):
        for j in range(n):
            if random() < p:
                a[count, 0] = i
                a[count, 1] = j
                count += 1
    assert a.shape[0] > count, 'Miscalculated bounds of max successful draws.'
    return a[0:count, :].copy()


@jit(nopython=True)
def random_labelgraph(n, l, p):  # noqa: E741
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """

    if n > 10:
        max_len = int(np.ceil(np.sum(n*(n-1)*p + 3*np.sqrt(n*(n-1)*p*(1-p)))))
        a = np.empty((max_len, 3), dtype=np.uint64)
    else:
        a = np.empty((n*(n-1)*l, 3), dtype=np.uint64)
    count = 0
    for i in range(l):
        for j in range(n):
            for k in range(n):
                if random() < p[i]:
                    a[count, 0] = i
                    a[count, 1] = j
                    a[count, 2] = k
                    count += 1
    assert a.shape[0] > count, 'Miscalculated bounds of max successful draws.'
    return a[0:count, :].copy()


# --------------- STRIPE METHODS ---------------
@jit(nopython=True)
def exp_edges_stripe_single_layer(z, out_strength, in_strength):
    exp_edges = 0
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(len(in_strength)):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                tmp = exp(z)*s_out*s_in
                if isinf(tmp):
                    exp_edges += 1
                else:
                    exp_edges += tmp / (1 + tmp)

    return exp_edges


@jit(nopython=True, parallel=True)
def exp_edges_stripe(z, out_strength, in_strength):
    """ Compute the expected number of edges for the stripe fitness model
    with one parameter controlling for the density for each label.
    """
    exp_edges = np.zeros(len(z), dtype=np.float64)
    num_it = max(out_strength.label.max(), in_strength.label.max()) + 1
    for i in prange(num_it):
        exp_edges[i] = exp_edges_stripe_single_layer(
            z[i],
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])
    return exp_edges


@jit(nopython=True)
def jac_stripe_single_layer(z, out_strength, in_strength):
    jac = 0
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(len(in_strength)):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                tmp = exp(z)*s_out*s_in
                if isinf(tmp):
                    jac += 0
                else:
                    jac += tmp / (1 + tmp)**2

    return jac


# --------------- OLD METHODS ---------------

@jit(nopython=True)
def iterative_stripe_one_z(z, out_str, in_str, L):
    """
    function computing the next iteration with
    the fixed point method for CSM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        s_out = out_str[i, 2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            s_in = in_str[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = s_out*s_in
                aux1 += aux2/(1+z*aux2)
    aux_z = L/aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_prime_stripe_one_z(z, out_str, in_str, L):
    """
    first derivative of the loglikelihood function
    of the CSM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        s_out = out_str[i, 2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            s_in = in_str[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = z*s_out*s_in
                aux1 += aux2/(1+aux2)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_hessian_stripe_one_z(z, out_str, in_str):
    """
    second derivative of the loglikelihood function
    of the CSM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        s_out = out_str[i, 2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            s_in = in_str[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = s_out*s_in
                aux1 -= aux2 / (1 + z*aux2)**2
    return aux1


@jit(nopython=True)
def iterative_block_one_z(z, out_str, in_str, L):
    """
    function computing the next iteration with
    the fixed point method for CBM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in) & (sect_out == sect_node_j) & (
                 sect_in == sect_node_i):
                aux2 = s_out*s_in
                aux1 += aux2/(1+z*aux2)
    aux_z = L/aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_prime_block_one_z(z, out_str, in_str, L):
    """
    first derivative of the loglikelihood function
    of the CBM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in) & (sect_out == sect_node_j) & (
                 sect_in == sect_node_i):
                aux2 = z*s_out*s_in
                aux1 += aux2/(1+aux2)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_hessian_block_one_z(z, out_str, in_str):
    """
    second derivative of the loglikelihood function
    of the CBM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in) & (sect_out == sect_node_j) & (
                 sect_in == sect_node_i):
                aux2 = s_out*s_in
                aux1 -= aux2 / (1 + z*aux2)**2
    return aux1


@jit(nopython=True)
def iterative_stripe_mult_z(z, out_strength, in_strength, L):
    """
    function computing the next iteration with
    the fixed point method for CSM-II model
    """
    aux1 = np.zeros(len(z))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = s_out*s_in
                aux1[sect_out] += aux2/(1+z[sect_out]*aux2)
    aux_z = np.zeros(len(z))
    for i in np.arange(aux1.shape[0]):
        if L[i]:
            aux_z[i] = L[i]/aux1[i]
    return aux_z


@jit(nopython=True)
def loglikelihood_prime_stripe_mult_z(z, out_strength, in_strength, L):
    """
    Function computing the first derivative of the
    loglikelihood function for the CSM-II model
    """
    aux_1 = np.zeros(L.shape[0])
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = z[sect_out]*s_out*s_in
                aux_1[sect_out] += aux2/(1+aux2)
    aux_z = L - aux_1
    return aux_z


@jit(nopython=True)
def loglikelihood_hessian_stripe_mult_z(z, out_strength, in_strength):
    """
    Function computing the second derivative of the
    loglikelihood function for the CSM-II model
    """
    aux_1 = np.zeros(len(z))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                aux2 = s_out*s_in
                aux_1[sect_out] -= aux2/((1+z[sect_out] * aux2)**2)
    return aux_1


@jit(nopython=True)
def iterative_block_mult_z(z, out_strength, in_strength, L):
    """
    function computing the next iteration with
    the fixed point method for CBM-II model
    """
    n_sector = int(np.sqrt(max(z.shape)))
    aux1 = np.zeros(shape=(n_sector, n_sector), dtype=np.float)
    z = np.reshape(z, newshape=(n_sector, n_sector))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                aux2 = s_out * s_in
                aux1[sect_out, sect_in] += aux2/(1+z[sect_out, sect_in]*aux2)
    aux_z = L/aux1
    return aux_z


# @jit(nopython=True, parallel=True)
def prob_matrix_stripe_one_z(out_strength, in_strength, z, N):
    """ Compute the probability matrix of the stripe fitness model given the
    single parameter z controlling for the density.
    """
    p = np.ones((N, N), dtype=np.float64)

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
                p[ind_out, ind_in] *= 1 - tmp2 / (1 + tmp2)

    return 1 - p


# @jit(nopython=True, parallel=True)
def prob_array_stripe_one_z(out_strength, in_strength, z, N, G):
    """ Compute the probability array of the stripe fitness model given the
    single parameter z controlling for the density.
    """
    p = np.zeros((N, N, G), dtype=np.float64)

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
                p[ind_out, ind_in, sect_out] = tmp2 / (1 + tmp2)

    return p


# @jit(forceobj=True, parallel=False)
def expected_links_stripe_one_z(out_strength, in_strength, z):
    """ Compute the expected number of links of the stripe fitness model
    given the single parameter z controlling for the density.
    """
    N = max(np.max(out_strength[:, 0]), np.max(in_strength[:, 0])) + 1
    return prob_matrix_stripe_one_z(out_strength, in_strength, z, int(N)).sum()


@jit(nopython=True)
def assign_weights_cimi_stripe_one_z(p, out_strength, in_strength,
                                     N, strengths_stripe, expected=True):
    """ Return the weighted adjacency matrix of the stripe fitness model
    with just one global parameter z controlling for the density.
    Depending on the value of "expected" the weighted adjacency matrix is the
    expceted one or an ensemble realisation.

    Parameters
    ----------
    p: scipy.sparse.matrix or np.ndarray or list of lists
        the binary probability matrix
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    strengths_stripe: np.ndarray
        strengths for stripe
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
                    W[ind_out, ind_in] = (tmp)/(tot_w)
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
                    if p[ind_out, ind_in] > np.random.random():
                        tot_w = strengths_stripe[sect_out]
                        tmp = s_out*s_in
                        W[ind_out, ind_in] = (tmp)/(tot_w*p[ind_out, ind_in])
    return W


@jit(nopython=True, parallel=True)
def prob_matrix_stripe_mult_z(out_strength, in_strength, z, N):
    """ Compute the probability matrix of the stripe fitness model with one
    parameter controlling for the density for each label.
    """
    p = np.ones((N, N), dtype=np.float64)

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
                p[ind_out, ind_in] *= 1 - tmp / (1 + tmp)
    return 1 - p


@jit(nopython=True, parallel=True)
def prob_array_stripe_mult_z(out_strength, in_strength, z, N, G):
    """ Compute the probability array of the stripe fitness model with one
    parameter controlling for the density for each label.
    """
    p = np.zeros((N, N, G), dtype=np.float64)

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
                tmp2 = z[sect_out]*tmp
                p[ind_out, ind_in, sect_out] = tmp2 / (1 + tmp2)

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


# @jit(forceobj=True, parallel=True)
def expected_links_block_one_z(out_strength, in_strength, z):
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
def assign_weights_cimi_block_one_z(p, out_strength, in_strength,
                                    N, strengths_block, expected=True):
    """Function returning the weighted adjacency matrix of the Cimi block
    model with just one global parameter z controlling for the density.
    Depending on the value of "expected" the weighted adjacency matrix can be
    the expceted one or just an ensemble realisation.

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
                    tot_w = strengths_block[sect_node_i, sect_node_j]
                    tmp = s_out*s_in
                    W[ind_out, ind_in] = (tmp)/(tot_w)
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
def vector_fitness_prob_array_block_mult_z(out_strength, in_strength, z, N):
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


# @jit(forceobj=True, parallel=True)
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
