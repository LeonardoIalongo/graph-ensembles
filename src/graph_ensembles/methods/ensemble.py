import numpy as np
import numpy.random as rng
from numba import jit
from math import ceil, sqrt, isinf, exp, log


# --------------- PROBABILITY FUNCTIONALS ---------------
@jit(nopython=True)
def p_fitness(d, x_i, x_j):
    tmp = d*x_i*x_j
    if isinf(tmp):
        return 1
    else:
        return tmp / (1 + tmp)


@jit(nopython=True)
def jac_fitness(d, x_i, x_j):
    tmp = x_i*x_j
    tmp1 = d*tmp
    if isinf(tmp1):
        return 0
    else:
        return tmp / (1 + tmp1)**2


@jit(nopython=True)
def p_invariant(d, x_i, x_j):
    tmp = d*x_i*x_j
    if isinf(tmp):
        return 1
    else:
        return 1 - exp(-tmp)


@jit(nopython=True)
def jac_invariant(d, x_i, x_j):
    tmp = x_i*x_j
    tmp1 = d*tmp
    if isinf(tmp1):
        return 0
    else:
        return tmp * exp(-tmp1)


@jit(nopython=True)
def p_fitness_alpha(d, x_i, x_j, a):
    tmp = d*(x_i**a)*(x_j**a)
    if isinf(tmp):
        res = 1.0
    else:
        res = tmp / (1 + tmp)
    return res


@jit(nopython=True)
def jac_fitness_alpha(d, x_i, x_j, a):
    tmp = (x_i**a)*(x_j**a)
    tmp1 = (1 + d*tmp)**2
    if isinf(tmp1):
        return 0, 0
    else:
        return tmp / tmp1, d*log(x_i*x_j)*tmp / tmp1


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

    if n > 100:
        if p > 0.95:
            a = np.empty((n*(n-1), n_clm), dtype=np.float64)
        else:
            max_len = int(ceil(n*(n-1)*p + 4*sqrt(n*(n-1)*p*(1-p))))
            a = np.empty((max_len, n_clm), dtype=np.float64)
    else:
        a = np.empty((n*(n-1), n_clm), dtype=np.float64)
        max_len = (n*(n-1))

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
    """ Generates a labelled edge list given the number of vertices and the
    probability p[l] of observing a link of a given label.
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


# --------------- FITNESS METHODS ---------------
@jit(nopython=True)
def fit_exp_degree_vertex(p_f, z, i, fit_i, fit_j):
    """ Compute the expected degree of the i-th vertex.
    """
    d = 0
    for j in np.arange(len(fit_j)):
        ind = fit_j[j].id
        val = fit_j[j].value
        if ind != i:
            d += p_f(z, fit_i, val)

    return d


@jit(nopython=True)
def fit_exp_edges(p_f, z, fit_out, fit_in):
    """ Compute the expected number of edges.
    """
    exp_edges = 0
    for i in np.arange(len(fit_out)):
        ind_out = fit_out[i].id
        v_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            ind_in = fit_in[j].id
            v_in = fit_in[j].value
            if ind_out != ind_in:
                exp_edges += p_f(z, v_out, v_in)

    return exp_edges


@jit(nopython=True)
def fit_exp_edges_jac(jac_f, z, fit_out, fit_in):
    """ Compute the Jacobian of the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    for i in np.arange(len(fit_out)):
        ind_out = fit_out[i].id
        s_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            ind_in = fit_in[j].id
            s_in = fit_in[j].value
            if ind_out != ind_in:
                jac += jac_f(z, s_out, s_in)

    return jac


@jit(nopython=True)
def fit_exp_edges_jac_alpha(jac_f, z, fit_out, fit_in):
    """ Compute the Jacobian of the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = np.zeros(2, dtype=np.float64)
    for i in np.arange(len(fit_out)):
        ind_out = fit_out[i].id
        s_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            ind_in = fit_in[j].id
            s_in = fit_in[j].value
            if ind_out != ind_in:
                res = jac_f(z, s_out, s_in)
                jac[0] += res[0]
                jac[1] += res[1]

    return jac


@jit(nopython=True)
def fit_exp_degree(p_f, z, fit_out, fit_in):
    """ Compute the expected in and out degree sequences.
    """
    exp_d_out = np.zeros(len(fit_out.id), dtype=np.float64)
    exp_d_in = np.zeros(len(fit_in.id), dtype=np.float64)
    for i in np.arange(len(fit_out)):
        ind_out = fit_out[i].id
        s_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            ind_in = fit_in[j].id
            s_in = fit_in[j].value
            if ind_out != ind_in:
                pij = p_f(z, s_out, s_in)
                exp_d_out[i] += pij
                exp_d_in[j] += pij

    d_out = []
    for i in range(len(exp_d_out)):
        d_out.append((fit_out[i].id, exp_d_out[i]))

    d_in = []
    for j in range(len(exp_d_in)):
        d_in.append((fit_in[j].id, exp_d_in[j]))

    return d_out, d_in


@jit(nopython=True)
def fit_f_jac(p_f, jac_f, z, fit_out, fit_in, n_edges):
    """ Compute the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    f = 0
    for i in np.arange(len(fit_out)):
        ind_out = fit_out[i].id
        s_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            ind_in = fit_in[j].id
            s_in = fit_in[j].value
            if ind_out != ind_in:
                f += p_f(z, s_out, s_in)
                jac += jac_f(z, s_out, s_in)

    return f - n_edges, jac


@jit(nopython=True)
def fit_iterative(z, out_strength, in_strength, n_edges):
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
                aux += tmp / (1 + z*tmp)

    return n_edges/aux


def fit_eq_constr_alpha(x, p_f, fit_out, fit_in, num_e):
    @jit(nopython=True)
    def f(d, x_i, x_j):
        return p_f(d, x_i, x_j, x[1])

    exp_e = fit_exp_edges(
         f,
         x[0],
         fit_out,
         fit_in)

    return np.array([exp_e - num_e], dtype=np.float64)


def fit_eq_jac_alpha(x, jac_f, fit_out, fit_in):
    @jit(nopython=True)
    def f(d, x_i, x_j):
        return jac_f(d, x_i, x_j, x[1])

    jac = fit_exp_edges_jac_alpha(f, x[0], fit_out, fit_in)

    return np.array([jac[0], jac[1]], dtype=np.float64)


def fit_ineq_constr_alpha(x, p_f, i, fit_i, fit_j):
    @jit(nopython=True)
    def f(d, x_i, x_j):
        return p_f(d, x_i, x_j, x[1])

    deg = fit_exp_degree_vertex(f, x[0], i, fit_i, fit_j)

    return np.array([deg], dtype=np.float64)


@jit(nopython=True)
def fit_ineq_jac_alpha(x, jac_f, i, fit_i, fit_j):
    jac = np.zeros(2, dtype=np.float64)
    for j in np.arange(len(fit_j)):
        ind_j = fit_j[j].id
        j_val = fit_j[j].value
        if i != ind_j:
            res = jac_f(x[0], fit_i, j_val, x[1])
            jac[0] += res[0]
            jac[1] += res[1]

    return jac


@jit(nopython=True)
def sample_stripe_single_layer(p_f, z, out_strength, in_strength, label):
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
                p = p_f(z, s_out, s_in)
                if rng.random() < p:
                    w = np.float64(rng.exponential(s_out*s_in/(s_tot*p)))
                    sample.append((label, ind_out, ind_in, w))

    return sample


# --------------- STRIPE METHODS ---------------
@jit(nopython=True)
def stripe_exp_edges(p_f, z, out_strength, in_strength, num_labels):
    """ Compute the expected number of edges with one parameter controlling
    for the density for each label.
    """
    exp_edges = np.zeros(len(z), dtype=np.float64)
    for i in range(num_labels):
        exp_edges[i] = fit_exp_edges(
            p_f,
            z[i],
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])
    return exp_edges


def stripe_exp_edges_alpha(
        p_f, alpha, z, out_strength, in_strength, num_labels):
    """ Compute the expected number of edges with one parameter controlling
    for the density for each label.
    """
    @jit(nopython=True)
    def f(d, x_i, x_j):
        return p_f(d, x_i, x_j, alpha[i])

    exp_edges = np.zeros(len(z), dtype=np.float64)
    for i in range(num_labels):
        exp_edges[i] = fit_exp_edges(
            f,
            z[i],
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])
    return exp_edges


@jit(nopython=True)
def stripe_pij(p_f, z, out_label, out_vals, in_label, in_vals):
    """ Computes the probability of observing a link between two nodes (i,j)
    over all labels as p_ij = 1 - prod_alpha(1 - p_ij^alpha).
    """
    p = 0
    i = 0
    j = 0
    while (i < len(out_label) and (j < len(in_label))):
        out_l = out_label[i]
        in_l = in_label[j]
        if out_l == in_l:
            p *= 1 - p_f(z[out_l], out_vals[i], in_vals[j])
            i += 1
            j += 1
        elif out_l < in_l:
            i += 1
        else:
            j += 1

    return 1 - p


@jit(nopython=True)
def stripe_exp_degree(p_f, z, s_out_i, s_out_j, s_out_w,
                      s_in_i, s_in_j, s_in_w, N):
    """ Calculate the expected degree for the stripe model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    label. s_out and s_in must be csr matrices.

    Arguments
    ----------
    z: float
        value of the density parameter z
    s_out_i: array
        indptr of the out_strength by label sparse csr matrix
    s_out_j: array
        indices of the out_strength by label sparse csr matrix
    s_out_w: array
        data of the out_strength by label sparse csr matrix
    s_in_i: array
        indices of the in_strength by label sparse csr matrix
    s_in_j: array
        indptr of the in_strength by label sparse csr matrix
    s_in_w: array
        data of the in_strength by label sparse csr matrix
    group_arr: array
        an array containing the label id for each vertex in order
    """
    out_degree = np.zeros(N, dtype=np.float64)
    in_degree = np.zeros(N, dtype=np.float64)

    # Iterate over vertex ids of the out strength and compute for each id the
    # expected degree
    for out_row in range(len(s_out_i)-1):
        # Get non-zero out strengths for vertex out_row of label out_label
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]

        if n == m:
            continue

        out_label = s_out_j[n:m]
        out_vals = s_out_w[n:m]

        for in_row in range(len(s_in_i)-1):
            # No self-loops
            if out_row == in_row:
                continue

            # Get non-zero in strengths for vertex out_row of label out_label
            r = s_in_i[in_row]
            s = s_in_i[in_row + 1]
            in_label = s_in_j[r:s]
            in_vals = s_in_w[r:s]

            if r == s:
                continue

            # Get pij
            pij = stripe_pij(p_f, z, out_label, out_vals, in_label, in_vals)
            out_degree[out_row] += pij
            in_degree[in_row] += pij

    return out_degree, in_degree


@jit(nopython=True)  # TODO: manca la label
def stripe_exp_degree_label(p_f, z, out_strength, in_strength, num_labels):
    """ Compute the expected degree by label for the stripe fitness model
    with one parameter controlling for the density for each label.
    """
    exp_d_out = []
    exp_d_in = []
    for i in range(num_labels):
        res = fit_exp_degree(
            p_f,
            z[i],
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])
        exp_d_out.extend(res[0])
        exp_d_in.extend(res[1])

    return exp_d_out, exp_d_in


@jit(nopython=True)
def stripe_sample(p_f, z, out_strength, in_strength, num_labels):
    """ Sample edges and weights from the stripe ensemble.
    """
    sample = []
    for i in range(num_labels):
        s_out = out_strength[out_strength.label == i]
        s_in = in_strength[in_strength.label == i]
        label = s_out[0].label
        sample.extend(
            sample_stripe_single_layer(p_f, z[i], s_out, s_in, label))

    return np.array(sample, dtype=np.float64)


def stripe_sample_alpha(p_f, z, alpha, out_strength, in_strength, num_labels):
    """ Sample edges and weights from the stripe ensemble.
    """
    @jit(nopython=True)
    def f(d, x_i, x_j):
        return p_f(d, x_i, x_j, alpha[i])

    sample = []
    for i in range(num_labels):
        s_out = out_strength[out_strength.label == i]
        s_in = in_strength[in_strength.label == i]
        label = s_out[0].label
        sample.extend(
            sample_stripe_single_layer(f, z[i], s_out, s_in, label))

    return np.array(sample, dtype=np.float64)


# --------------- BLOCK METHODS ---------------
@jit(nopython=True)
def block_exp_vertex_degree(p_f, z, out_g, out_vals, in_gj, in_vals):
    d = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                d += p_f(z, out_vals[i], in_vals[j])

    return d


@jit(nopython=True)
def f_jac_block_i(p_f, jac_f, z, out_g, out_vals, in_gj, in_vals):
    f = 0
    jac = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                f += p_f(z, out_vals[i], in_vals[j])
                jac += jac_f(z, out_vals[i], in_vals[j])
    return f, jac


@jit(nopython=True)
def block_exp_num_edges(p_f, z, s_out_i, s_out_j, s_out_w,
                        s_in_i, s_in_j, s_in_w, group_arr):
    """ Calculate the expected number of edges for the block model.

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
        num += block_exp_vertex_degree(p_f, z, out_g, out_vals, in_gj, in_vals)

    return num


@jit(nopython=True)
def block_exp_out_degree(p_f, z, s_out_i, s_out_j, s_out_w,
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
            p_f, z, out_g, out_vals, in_gj, in_vals)

    return out_degree


@jit(nopython=True)
def f_jac_block(p_f, jac_f, z, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j,
                s_in_w, group_arr, num_e):
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
        res = f_jac_block_i(p_f, jac_f, z, out_g, out_vals, in_gj, in_vals)
        f += res[0]
        jac += res[1]

    return f, jac


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
                    aux += tmp / (1 + z*tmp)

    return num_e/aux


@jit(nopython=True)
def sample_block_vertex(p_f, z, out_i, out_g, out_vals,
                        in_j, in_gj, in_vals, s_tot):
    """ Sample edges going out from a single vertex.
    """
    sample = []
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                p = p_f(z, out_vals[i], in_vals[j])
                if rng.random() < p:
                    w = np.float64(
                        rng.exponential(out_vals[i]*in_vals[j]/(s_tot*p)))
                    sample.append((out_i, in_j[j], w))

    return sample


@jit(nopython=True)
def block_sample(p_f, z, s_out_i, s_out_j, s_out_w,
                 s_in_i, s_in_j, s_in_w, group_arr):
    """ Sample from block model.
    """
    s_tot = np.sum(s_out_w)
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
            sample_block_vertex(p_f, z, out_row, out_g, out_vals,
                                in_j[notself], in_gj, in_vals, s_tot))

    return np.array(sample, dtype=np.float64)
