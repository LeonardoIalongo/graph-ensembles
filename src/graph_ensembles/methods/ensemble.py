import numpy as np
import numpy.random as rng
from numba import jit
from math import ceil, sqrt, isinf, exp, log


# --------------- RANDOM GRAPH METHODS ---------------
@jit(nopython=True)
def random_graph(n, p, q=None, discrete_weights=False):
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """
    if q is None:
        n_clm = 2
    else:
        n_clm = 3

    if n > 10:
        if p > 0.95:
            a = np.empty((n*(n-1), n_clm), dtype=np.float64)
        else:
            max_len = int(ceil(n*(n-1)*p + 4*sqrt(n*(n-1)*p*(1-p))))
            a = np.empty((max_len, n_clm), dtype=np.float64)
    else:
        a = np.empty((n*(n-1), n_clm), dtype=np.float64)

    count = 0
    msg = 'Miscalculated bounds of max successful draws.'
    for i in range(n):
        for j in range(n):
            if i != j:
                if rng.random() < p:
                    assert count < max_len, msg
                    a[count, 0] = i
                    a[count, 1] = j
                    if q is not None:
                        if discrete_weights:
                            w = rng.geometric(1 - q)
                        else:
                            w = rng.exponential(1/q)
                        a[count, 2] = w
                    count += 1

    return a[0:count, :].copy()


@jit(nopython=True)
def random_labelgraph(n, l, p, q=None, discrete_weights=False):  # noqa: E741
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """
    if q is None:
        n_clm = 3
    else:
        n_clm = 4

    if n > 10:
        p_aux = p.copy()
        p_aux[p_aux > 0.95] = 1
        max_len = int(np.ceil(np.sum(
            n*(n-1)*p_aux + 4*np.sqrt(n*(n-1)*p_aux*(1-p_aux)))))
        a = np.empty((max_len, n_clm), dtype=np.float64)
    else:
        a = np.empty((n*(n-1)*l, n_clm), dtype=np.float64)

    count = 0
    msg = 'Miscalculated bounds of max successful draws.'
    for i in range(l):
        for j in range(n):
            for k in range(n):
                if j != k:
                    if rng.random() < p[i]:
                        assert count < max_len, msg
                        a[count, 0] = i
                        a[count, 1] = j
                        a[count, 2] = k
                        if q is not None:
                            if discrete_weights:
                                w = rng.geometric(1 - q[i])
                            else:
                                w = rng.exponential(1/q[i])
                            a[count, 3] = w
                        count += 1

    return a[0:count, :].copy()


# --------------- STRIPE METHODS ---------------
@jit(nopython=True)
def exp_edges_stripe_single_layer(z, out_strength, in_strength):
    """ Compute the expected number of edges for a single label of the stripe
    model.
    """
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


@jit(nopython=True)
def exp_edges_stripe(z, out_strength, in_strength, num_labels):
    """ Compute the expected number of edges for the stripe fitness model
    with one parameter controlling for the density for each label.
    """
    exp_edges = np.zeros(len(z), dtype=np.float64)
    for i in range(num_labels):
        exp_edges[i] = exp_edges_stripe_single_layer(
            log(z[i]),
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])
    return exp_edges


@jit(nopython=True)
def f_jac_stripe_single_layer(z, out_strength, in_strength, n_edges):
    """ Compute the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    f = 0
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(len(in_strength)):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                tmp = exp(z)*s_out*s_in
                if isinf(tmp):
                    f += 1
                    jac += 0
                else:
                    f += tmp / (1 + tmp)
                    jac += tmp / (1 + tmp)**2

    return f - n_edges, jac


@jit(nopython=True)
def stripe_newton_init(out_strength, in_strength, n_e, steps):
    """ Compute initial conditions for the stripe solvers.
    """
    z = 0
    for n in range(steps):
        jac = 0
        f = 0
        for i in np.arange(len(out_strength)):
            ind_out = out_strength[i].id
            s_out = out_strength[i].value
            for j in np.arange(len(in_strength)):
                ind_in = in_strength[j].id
                s_in = in_strength[j].value
                if ind_out != ind_in:
                    tmp = s_out*s_in
                    tmp1 = z*tmp
                    jac += tmp / (1 + tmp1)**2
                    f += z*tmp1 / (1 + z*tmp1)
        z = z - (f-n_e)/jac

    return z


@jit(nopython=True)
def iterative_stripe_single_layer(z, out_strength, in_strength, n_edges):
    """ Compute the next iteration of the fixed point method for a single
    label of the stripe model.
    """
    aux = 0
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(in_strength.shape[0]):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                tmp = s_out*s_in
                aux += tmp / (1 + exp(z)*tmp)

    return log(n_edges/aux)


@jit(nopython=True)
def sample_stripe_single_layer(z, out_strength, in_strength, label):
    """ Compute the expected number of edges for a single label of the stripe
    model.
    """
    s_tot = np.sum(out_strength.value)
    msg = 'Sum of in/out strengths not the same.'
    assert np.abs(1 - np.sum(in_strength.value)/s_tot) < 1e-6, msg
    sample = []
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(len(in_strength)):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                tmp = z*s_out*s_in
                p = tmp / (1 + tmp)
                if rng.random() < p:
                    w = np.float64(rng.exponential(s_out*s_in/(s_tot*p)))
                    sample.append((label, ind_out, ind_in, w))

    return sample


@jit(nopython=True)
def stripe_sample(z, out_strength, in_strength, num_labels):
    """ Sample edges and weights from the stripe ensemble.
    """
    sample = []
    for i in range(num_labels):
        s_out = out_strength[out_strength.label == i]
        s_in = in_strength[in_strength.label == i]
        label = s_out[0].label
        sample.extend(sample_stripe_single_layer(z[i], s_out, s_in, label))

    return np.array(sample, dtype=np.float64)


# --------------- BLOCK METHODS ---------------
@jit(nopython=True)
def block_exp_vertex_degree(z, out_g, out_vals, in_gj, in_vals):
    d = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                tmp = z*out_vals[i]*in_vals[j]
                d += tmp / (1 + tmp)

    return d


@jit(nopython=True)
def f_jac_block_i(z, out_g, out_vals, in_gj, in_vals):
    f = 0
    jac = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                tmp = exp(z)*out_vals[i]*in_vals[j]
                if isinf(tmp):
                    f += 1
                    jac += 0
                else:
                    f += tmp / (1 + tmp)
                    jac += tmp / (1 + tmp)**2
    return f, jac


@jit(nopython=True)
def block_exp_num_edges(z, s_out_i, s_out_j, s_out_w,
                        s_in_i, s_in_j, s_in_w, group_arr):
    """ Calculate the expecte number of edges for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    z: float
        value of the density parameter z
    s_out_i: array
        indptr of the out_strength by group sparse csr matrix
    s_out_j: array
        indices of the out_strength by group sparse csr matrix
    s_out_w: array
        data of the out_strength by group sparse csr matrix
    s_in_i: array
        indices of the in_strength by group sparse csc matrix
    s_in_j: array
        indptr of the in_strength by group sparse csc matrix
    s_in_w: array
        data of the in_strength by group sparse csc matrix
    group_arr: array
        an array containing the group id for each vertex in order
    """
    num = 0

    # Iterate over vertex ids of the out strength and compute for each id the
    # expected degree
    for out_row in range(len(s_out_i)-1):
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]
        r = s_in_j[group_arr[out_row]]
        s = s_in_j[group_arr[out_row]+1]

        # Ensure at least one element exists
        if (n == m) or (r == s):
            continue

        # Get non-zero out strengths for vertex out_row towards groups out_g
        out_g = s_out_j[n:m]
        out_vals = s_out_w[n:m]

        # Get indices of non-zero in strengths from group of vertex out_row
        in_j = s_in_i[r:s]

        # Remove self loops
        notself = in_j != out_row

        # Ensure at least one element remains
        if np.sum(notself) == 0:
            continue

        # Get groups corresponding to the in_j values
        in_gj = group_arr[in_j][notself]
        in_vals = s_in_w[r:s][notself]
        num += block_exp_vertex_degree(z, out_g, out_vals, in_gj, in_vals)

    return num


@jit(nopython=True)
def block_exp_out_degree(z, s_out_i, s_out_j, s_out_w,
                         s_in_i, s_in_j, s_in_w, group_arr):
    """ Calculate the expected out degree for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    z: float
        value of the density parameter z
    s_out_i: array
        indptr of the out_strength by group sparse csr matrix
    s_out_j: array
        indices of the out_strength by group sparse csr matrix
    s_out_w: array
        data of the out_strength by group sparse csr matrix
    s_in_i: array
        indices of the in_strength by group sparse csc matrix
    s_in_j: array
        indptr of the in_strength by group sparse csc matrix
    s_in_w: array
        data of the in_strength by group sparse csc matrix
    group_arr: array
        an array containing the group id for each vertex in order
    """
    out_degree = np.zeros(len(group_arr), dtype=np.int64)

    # Iterate over vertex ids of the out strength and compute for each id the
    # expected degree
    for out_row in range(len(s_out_i)-1):
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]
        r = s_in_j[group_arr[out_row]]
        s = s_in_j[group_arr[out_row]+1]

        # Ensure at least one element exists
        if (n == m) or (r == s):
            continue

        # Get non-zero out strengths for vertex out_row towards groups out_g
        out_g = s_out_j[n:m]
        out_vals = s_out_w[n:m]

        # Get indices of non-zero in strengths from group of vertex out_row
        in_j = s_in_i[r:s]

        # Remove self loops
        notself = in_j != out_row

        # Ensure at least one element remains
        if np.sum(notself) == 0:
            continue

        # Get groups corresponding to the in_j values
        in_gj = group_arr[in_j][notself]
        in_vals = s_in_w[r:s][notself]
        out_degree[out_row] = block_exp_vertex_degree(
            z, out_g, out_vals, in_gj, in_vals)

    return out_degree


@jit(nopython=True)
def f_jac_block(z, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j, s_in_w,
                group_arr, num_e):
    """ Calculate the objective function and its Jacobian for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    z: float
        value of the density parameter z
    s_out_i: array
        indptr of the out_strength by group sparse csr matrix
    s_out_j: array
        indices of the out_strength by group sparse csr matrix
    s_out_w: array
        data of the out_strength by group sparse csr matrix
    s_in_i: array
        indices of the in_strength by group sparse csc matrix
    s_in_j: array
        indptr of the in_strength by group sparse csc matrix
    s_in_w: array
        data of the in_strength by group sparse csc matrix
    group_arr: array
        an array containing the group id for each vertex in order
    """
    f = -num_e
    jac = 0

    # Iterate over vertex ids of the out strength and compute for each id the
    # expected degree
    for out_row in range(len(s_out_i)-1):
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]
        r = s_in_j[group_arr[out_row]]
        s = s_in_j[group_arr[out_row]+1]

        # Ensure at least one element exists
        if (n == m) or (r == s):
            continue

        # Get non-zero out strengths for vertex out_row towards groups out_g
        out_g = s_out_j[n:m]
        out_vals = s_out_w[n:m]

        # Get indices of non-zero in strengths from group of vertex out_row
        in_j = s_in_i[r:s]

        # Remove self loops
        notself = in_j != out_row

        # Ensure at least one element remains
        if np.sum(notself) == 0:
            continue

        # Get groups corresponding to the in_j values
        in_gj = group_arr[in_j][notself]
        in_vals = s_in_w[r:s][notself]
        res = f_jac_block_i(z, out_g, out_vals, in_gj, in_vals)
        f += res[0]
        jac += res[1]

    return f, jac


@jit(nopython=True)
def block_newton_init(s_out_i, s_out_j, s_out_w, s_in_i, s_in_j, s_in_w,
                      group_arr, n_e, steps):
    """ Compute initial conditions for the block model solvers.
    """
    z = 0
    for n in range(steps):
        jac = 0
        f = 0
        for out_row in range(len(s_out_i)-1):
            n = s_out_i[out_row]
            m = s_out_i[out_row + 1]
            r = s_in_j[group_arr[out_row]]
            s = s_in_j[group_arr[out_row]+1]

            if (n == m) or (r == s):
                continue

            out_g = s_out_j[n:m]
            out_vals = s_out_w[n:m]
            in_j = s_in_i[r:s]
            notself = in_j != out_row

            if np.sum(notself) == 0:
                continue

            in_gj = group_arr[in_j][notself]
            in_vals = s_in_w[r:s][notself]

            for i in range(len(out_g)):
                for j in range(len(in_gj)):
                    if out_g[i] == in_gj[j]:
                        tmp = out_vals[i]*in_vals[j]
                        tmp1 = z*tmp
                        jac += tmp / (1 + tmp1)**2
                        f += z*tmp1 / (1 + z*tmp1)

        z = z - (f-n_e)/jac

    return z


@jit(nopython=True)
def iterative_block(z, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j, s_in_w,
                    group_arr, num_e):
    """ Compute the next iteration of the fixed point method of the block model.
    """
    aux = 0
    for out_row in range(len(s_out_i)-1):
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]
        r = s_in_j[group_arr[out_row]]
        s = s_in_j[group_arr[out_row]+1]

        if (n == m) or (r == s):
            continue

        out_g = s_out_j[n:m]
        out_vals = s_out_w[n:m]
        in_j = s_in_i[r:s]
        notself = in_j != out_row

        if np.sum(notself) == 0:
            continue

        in_gj = group_arr[in_j][notself]
        in_vals = s_in_w[r:s][notself]

        for i in range(len(out_g)):
            for j in range(len(in_gj)):
                if out_g[i] == in_gj[j]:
                    tmp = out_vals[i]*in_vals[j]
                    aux += tmp / (1 + exp(z)*tmp)

    return log(num_e/aux)


def sample_block_vertex(z, out_i, out_g, out_vals,
                        in_j, in_gj, in_vals, s_tot):
    """ Sample edges going out from a single vertex.
    """
    sample = []
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                tmp = out_vals[i]*in_vals[j]
                tmp1 = z*tmp
                p = tmp1 / (1 + tmp1)
                if rng.random() < p:
                    w = np.float64(rng.exponential(tmp/(s_tot*p)))
                    sample.append((out_i, in_j[j], w))

    return sample


def block_sample(z, s_out_i, s_out_j, s_out_w,
                 s_in_i, s_in_j, s_in_w, group_arr):
    """ Sample from block model.
    """
    s_tot = np.sum(s_out_w)
    assert np.isclose(s_tot, np.sum(s_in_w)), 'Total strengths not matching.'
    sample = []
    for out_row in range(len(s_out_i)-1):
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]
        r = s_in_j[group_arr[out_row]]
        s = s_in_j[group_arr[out_row]+1]

        # Ensure at least one element exists
        if (n == m) or (r == s):
            continue

        # Get non-zero out strengths for vertex out_row towards groups out_g
        out_g = s_out_j[n:m]
        out_vals = s_out_w[n:m]

        # Get indices of non-zero in strengths from group of vertex out_row
        in_j = s_in_i[r:s]

        # Remove self loops
        notself = in_j != out_row

        # Ensure at least one element remains
        if np.sum(notself) == 0:
            continue

        # Get groups corresponding to the in_j values
        in_gj = group_arr[in_j][notself]
        in_vals = s_in_w[r:s][notself]
        sample.extend(
            sample_block_vertex(z, out_row, out_g, out_vals,
                                in_j[notself], in_gj, in_vals, s_tot))

    return np.array(sample, dtype=np.float64)


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


@jit(nopython=True)
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
        for j in range(in_strength.shape[0]):
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


# @jit(forceobj=True)
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
        for j in range(in_strength.shape[0]):
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


@jit(nopython=True)
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
        for j in range(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


# @jit(forceobj=True)
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
        for j in range(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[sect_out, sect_in] += tmp2 / (1 + tmp2)
    return np.reshape(p, -1)
