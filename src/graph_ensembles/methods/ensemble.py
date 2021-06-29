import numpy as np
import numpy.random as rng
from numba import jit
from math import ceil, sqrt, isinf, exp, log


# --------------- PROBABILITY FUNCTIONALS ---------------
@jit(nopython=True)
def p_fitness(param, x_i, x_j):
    tmp = param[0]*x_i*x_j
    if isinf(tmp):
        return 1
    else:
        return tmp / (1 + tmp)


@jit(nopython=True)
def jac_fitness(param, x_i, x_j):
    tmp = x_i*x_j
    tmp1 = param[0]*tmp
    if isinf(tmp1):
        return 0
    else:
        return tmp / (1 + tmp1)**2


@jit(nopython=True)
def p_invariant(param, x_i, x_j):
    tmp = param[0]*x_i*x_j
    if isinf(tmp):
        return 1
    else:
        return 1 - exp(-tmp)


@jit(nopython=True)
def jac_invariant(param, x_i, x_j):
    tmp = x_i*x_j
    tmp1 = param[0]*tmp
    if isinf(tmp1):
        return 0
    else:
        return tmp * exp(-tmp1)


@jit(nopython=True)
def p_fitness_alpha(param, x_i, x_j):
    tmp = param[0]*((x_i*x_j)**param[1])
    if isinf(tmp):
        res = 1.0
    else:
        res = tmp / (1 + tmp)
    return res


@jit(nopython=True)
def jac_fitness_alpha(param, x_i, x_j):
    tmp = (x_i*x_j)**param[1]
    tmp1 = (1 + param[0]*tmp)**2
    if isinf(tmp1):
        return 0, 0
    elif tmp == 0:
        return 0, 0
    else:
        return tmp / tmp1, param[0]*log(x_i*x_j)*tmp / tmp1


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
def fit_exp_degree_vertex(p_f, param, i, fit_i, fit_j):
    """ Compute the expected degree of the i-th vertex.
    """
    d = 0
    for j in np.arange(len(fit_j)):
        val = fit_j[j]
        if j != i:
            d += p_f(param, fit_i, val)

    return d


@jit(nopython=True)
def fit_exp_edges(p_f, param, fit_out, fit_in):
    """ Compute the expected number of edges.
    """
    exp_edges = 0
    for i in np.arange(len(fit_out)):
        v_out = fit_out[i]
        for j in np.arange(len(fit_in)):
            v_in = fit_in[j]
            if i != j:
                exp_edges += p_f(param, v_out, v_in)

    return exp_edges


@jit(nopython=True)
def fit_exp_edges_jac(jac_f, param, fit_out, fit_in):
    """ Compute the Jacobian of the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    for i in np.arange(len(fit_out)):
        s_out = fit_out[i]
        for j in np.arange(len(fit_in)):
            s_in = fit_in[j]
            if i != j:
                jac = jac_f(param, s_out, s_in)

    return jac


@jit(nopython=True)
def fit_exp_edges_jac_alpha(jac_f, param, fit_out, fit_in):
    """ Compute the Jacobian of the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = np.zeros(2, dtype=np.float64)
    for i in np.arange(len(fit_out)):
        s_out = fit_out[i]
        for j in np.arange(len(fit_in)):
            s_in = fit_in[j]
            if i != j:
                res = jac_f(param, s_out, s_in)
                jac[0] += res[0]
                jac[1] += res[1]

    return jac


@jit(nopython=True)
def fit_exp_degree(p_f, param, fit_out, fit_in):
    """ Compute the expected in and out degree sequences.
    """
    exp_d = np.zeros(len(fit_out), dtype=np.float64)
    exp_d_out = np.zeros(len(fit_out), dtype=np.float64)
    exp_d_in = np.zeros(len(fit_in), dtype=np.float64)
    for i in np.arange(len(fit_out)):
        s_out_i = fit_out[i]
        s_in_i = fit_in[i]
        for j in np.arange(i + 1, len(fit_in)):
            s_out_j = fit_out[j]
            s_in_j = fit_in[j]
            pij = p_f(param, s_out_i, s_in_j)
            pji = p_f(param, s_out_j, s_in_i)
            p = 1 - (1 - pij)*(1 - pji)
            exp_d[i] += p
            exp_d[j] += p
            exp_d_out[i] += pij
            exp_d_out[j] += pji
            exp_d_in[j] += pij
            exp_d_in[i] += pji

    return exp_d, exp_d_out, exp_d_in


@jit(nopython=True)
def fit_f_jac(p_f, jac_f, param, fit_out, fit_in, n_edges):
    """ Compute the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    f = 0
    for i in np.arange(len(fit_out)):
        s_out = fit_out[i]
        for j in np.arange(len(fit_in)):
            s_in = fit_in[j]
            if i != j:
                f += p_f(param, s_out, s_in)
                jac += jac_f(param, s_out, s_in)

    return f - n_edges, jac


@jit(nopython=True)
def fit_iterative(param, out_strength, in_strength, n_edges):
    """ Compute the next iteration of the fixed point method for a single
    label of the stripe model.
    """
    aux = 0
    for i in np.arange(len(out_strength)):
        s_out = out_strength[i]
        for j in np.arange(in_strength.shape[0]):
            s_in = in_strength[j]
            if i != j:
                tmp = s_out*s_in
                aux += tmp / (1 + param[0]*tmp)

    return n_edges/aux


@jit(nopython=True)
def fit_eq_constr_alpha(x, p_f, fit_out, fit_in, num_e):
    exp_e = fit_exp_edges(p_f, x, fit_out, fit_in)
    return np.array([exp_e - num_e], dtype=np.float64)


@jit(nopython=True)
def fit_eq_jac_alpha(x, jac_f, fit_out, fit_in):
    jac = fit_exp_edges_jac_alpha(jac_f, x, fit_out, fit_in)
    return np.array([jac[0], jac[1]], dtype=np.float64)


@jit(nopython=True)
def fit_ineq_constr_alpha(x, p_f, i, fit_i, fit_j):
    deg = fit_exp_degree_vertex(p_f, x, i, fit_i, fit_j)
    return deg - 1


@jit(nopython=True)
def fit_ineq_jac_alpha(x, jac_f, i, fit_i, fit_j):
    jac = np.zeros(2, dtype=np.float64)
    for j in np.arange(len(fit_j)):
        j_val = fit_j[j]
        if i != j:
            res = jac_f(x, fit_i, j_val)
            jac[0] += res[0]
            jac[1] += res[1]

    return jac


@jit(nopython=True)
def fit_av_nn_prop(p_f, param, fit_out, fit_in, prop, ndir='out'):
    """ Compute the expected number of edges.
    """
    av_nn = np.zeros(prop.shape, dtype=np.float64)
    for i in np.arange(len(fit_out)):
        s_out_i = fit_out[i]
        s_in_i = fit_in[i]
        for j in np.arange(i + 1, len(fit_in)):
            s_out_j = fit_out[j]
            s_in_j = fit_in[j]
            pij = p_f(param, s_out_i, s_in_j)
            pji = p_f(param, s_out_j, s_in_i)
            if ndir == 'out':
                av_nn[i] += pij*prop[j]
                av_nn[j] += pji*prop[i]
            elif ndir == 'in':
                av_nn[i] += pji*prop[j]
                av_nn[j] += pij*prop[i]
            elif ndir == 'out-in':
                p = 1 - (1 - pij)*(1 - pji)
                av_nn[i] += p*prop[j]
                av_nn[j] += p*prop[i]
            else:
                raise ValueError('Direction of neighbourhood not right.')

    return av_nn


@jit(nopython=True)
def fit_likelihood(adj_i, adj_j, p_f, param, out_strength, in_strength, lgs):
    """ Compute the binary log likelihood of a graph given the fitted model.
    """
    if lgs:
        like = 0
    else:
        like = 1

    for i in np.arange(len(out_strength)):
        s_out = out_strength[i]
        n = adj_i[i]
        m = adj_i[i+1]
        j_list = adj_j[n:m]
        for j in np.arange(len(in_strength)):
            s_in = in_strength[j]
            if i != j:
                p = p_f(param, s_out, s_in)
            else:
                p = 0

            # Check if link exists
            if j in j_list:
                if lgs:
                    like += log(p)
                else:
                    like *= p
            else:
                if lgs:
                    like += log(1 - p)
                else:
                    like *= 1 - p
    if lgs:
        return like
    else:
        return log(like)


@jit(nopython=True)
def fit_sample(p_f, param, out_strength, in_strength):
    """ Sample from the fitness model ensemble.
    """
    s_tot = np.sum(out_strength)
    msg = 'Sum of in/out strengths not the same.'
    assert np.abs(1 - np.sum(in_strength)/s_tot) < 1e-6, msg
    sample = []
    for i in np.arange(len(out_strength)):
        s_out = out_strength[i]
        for j in np.arange(len(in_strength)):
            s_in = in_strength[j]
            if i != j:
                p = p_f(param, s_out, s_in)
                if rng.random() < p:
                    w = np.float64(rng.exponential(s_out*s_in/(s_tot*p)))
                    sample.append((i, j, w))

    return sample


# --------------- LAYER FITNESS METHODS ---------------
@jit(nopython=True)
def layer_exp_edges(p_f, param, fit_out, fit_in):
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
                exp_edges += p_f(param, v_out, v_in)

    return exp_edges


@jit(nopython=True)
def layer_f_jac(p_f, jac_f, param, fit_out, fit_in, n_edges):
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
                f += p_f(param, s_out, s_in)
                jac += jac_f(param, s_out, s_in)

    return f - n_edges, jac


@jit(nopython=True)
def layer_iterative(param, out_strength, in_strength, n_edges):
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
                aux += tmp / (1 + param[0]*tmp)

    return n_edges/aux


@jit(nopython=True)
def layer_exp_degree(p_f, param, fit_out, fit_in, label):
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
                pij = p_f(param, s_out, s_in)
                exp_d_out[i] += pij
                exp_d_in[j] += pij

    d_out = []
    for i in range(len(exp_d_out)):
        d_out.append((label, fit_out[i].id, exp_d_out[i]))

    d_in = []
    for j in range(len(exp_d_in)):
        d_in.append((label, fit_in[j].id, exp_d_in[j]))

    return d_out, d_in


@jit(nopython=True)
def layer_eq_constr_alpha(x, p_f, fit_out, fit_in, num_e):
    exp_e = layer_exp_edges(p_f, x, fit_out, fit_in)
    return np.array([exp_e - num_e], dtype=np.float64)


@jit(nopython=True)
def layer_exp_edges_jac_alpha(jac_f, param, fit_out, fit_in):
    """ Compute the Jacobian of the objective function of the newton solver 
    and its derivative for a single label of the stripe model.
    """
    jac = np.zeros(2, dtype=np.float64)
    for i in np.arange(len(fit_out)):
        i_out = fit_out[i].id
        s_out = fit_out[i].value
        for j in np.arange(len(fit_in)):
            j_in = fit_in[j].id
            s_in = fit_in[j].value
            if i_out != j_in:
                res = jac_f(param, s_out, s_in)
                jac[0] += res[0]
                jac[1] += res[1]

    return jac


@jit(nopython=True)
def layer_eq_jac_alpha(x, jac_f, fit_out, fit_in):
    jac = layer_exp_edges_jac_alpha(jac_f, x, fit_out, fit_in)
    return np.array([jac[0], jac[1]], dtype=np.float64)


@jit(nopython=True)
def layer_exp_degree_vertex(p_f, param, i, fit_i, fit_j):
    """ Compute the expected degree of the i-th vertex.
    """
    d = 0
    for j in np.arange(len(fit_j)):
        ind = fit_j[j].id
        val = fit_j[j].value
        if ind != i:
            d += p_f(param, fit_i, val)

    return d


@jit(nopython=True)
def layer_ineq_constr_alpha(x, p_f, i, fit_i, fit_j):
    deg = layer_exp_degree_vertex(p_f, x, i, fit_i, fit_j)
    return deg - 1


@jit(nopython=True)
def layer_ineq_jac_alpha(x, jac_f, i, fit_i, fit_j):
    jac = np.zeros(2, dtype=np.float64)
    for j in np.arange(len(fit_j)):
        ind_j = fit_j[j].id
        j_val = fit_j[j].value
        if i != ind_j:
            res = jac_f(x, fit_i, j_val)
            jac[0] += res[0]
            jac[1] += res[1]

    return jac


@jit(nopython=True)
def layer_sample(p_f, param, out_strength, in_strength, label):
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
                p = p_f(param, s_out, s_in)
                if rng.random() < p:
                    w = np.float64(rng.exponential(s_out*s_in/(s_tot*p)))
                    sample.append((label, ind_out, ind_in, w))

    return sample


# --------------- STRIPE METHODS ---------------
@jit(nopython=True)
def stripe_pij(p_f, param, out_label, out_vals, in_label, in_vals, per_label):
    """ Computes the probability of observing a link between two nodes (i,j)
    over all labels as p_ij = 1 - prod_alpha(1 - p_ij^alpha).
    """
    p = 1
    i = 0
    j = 0
    while (i < len(out_label) and (j < len(in_label))):
        out_l = out_label[i]
        in_l = in_label[j]
        if out_l == in_l:
            if per_label:
                p *= 1 - p_f(param[:, out_l], out_vals[i], in_vals[j])
            else:
                p *= 1 - p_f(param[:, 0], out_vals[i], in_vals[j])
            i += 1
            j += 1
        elif out_l < in_l:
            i += 1
        else:
            j += 1

    return 1 - p


@jit(nopython=True)
def stripe_pij_f_jac(p_f, jac_f, param, out_label, out_vals,
                     in_label, in_vals):
    """ Computes the probability of observing a link between two nodes (i,j)
    over all labels as p_ij = 1 - prod_alpha(1 - p_ij^alpha).
    """
    max_a = max(len(out_label), len(in_label))
    p = np.zeros(max_a, dtype=np.float64)
    p_jac = np.zeros(max_a, dtype=np.float64)
    i = 0
    j = 0
    a = 0
    while (i < len(out_label) and (j < len(in_label))):
        out_l = out_label[i]
        in_l = in_label[j]
        if out_l == in_l:
            p[a] = p_f(param, out_vals[i], in_vals[j])
            p_jac[a] = jac_f(param, out_vals[i], in_vals[j])
            i += 1
            j += 1
            a += 1
        elif out_l < in_l:
            i += 1
        else:
            j += 1

    if np.any(p == 1):
        return 1, 0
    else:
        tmp = np.prod(1 - p)
        pij_jac = tmp*np.sum(p_jac / (1 - p))
        return 1 - tmp, pij_jac


@jit(nopython=True)
def stripe_pij_alpha(p_f, param, out_label, out_vals, in_label, in_vals):
    """ Computes the probability of observing a link between two nodes (i,j)
    over all labels as p_ij = 1 - prod_alpha(1 - p_ij^alpha).
    """
    p = 1
    i = 0
    j = 0
    while (i < len(out_label) and (j < len(in_label))):
        out_l = out_label[i]
        in_l = in_label[j]
        if out_l == in_l:
            p *= 1 - p_f(param, out_vals[i], in_vals[j])
            i += 1
            j += 1
        elif out_l < in_l:
            i += 1
        else:
            j += 1

    return 1 - p


@jit(nopython=True)
def stripe_pij_jac_alpha(p_f, jac_f, param, out_label, out_vals,
                         in_label, in_vals):
    """ Computes the derivative of the probability of observing a link between
    two nodes (i,j) over all labels.
    """
    max_a = max(len(out_label), len(in_label))
    p = np.zeros(max_a, dtype=np.float64)
    p_jac = np.zeros((2, max_a), dtype=np.float64)
    i = 0
    j = 0
    a = 0
    while (i < len(out_label) and (j < len(in_label))):
        out_l = out_label[i]
        in_l = in_label[j]
        if out_l == in_l:
            p[a] = p_f(param, out_vals[i], in_vals[j])
            tmp = jac_f(param, out_vals[i], in_vals[j])
            p_jac[0, a] = tmp[0]
            p_jac[1, a] = tmp[1]
            i += 1
            j += 1
            a += 1
        elif out_l < in_l:
            i += 1
        else:
            j += 1

    if np.any(p == 1):
        return np.zeros(2, dtype=np.float64)
    else:
        tmp = np.prod(1 - p)
        pij_jac = tmp*np.sum(p_jac / (1 - p), axis=1)
        return pij_jac


@jit(nopython=True)
def stripe_exp_degree_vertex(p_f, param, i_id, i_label, i_val, 
                             j_ptr, j_labels, j_vals):
    """ Calculate the expected degree for the i-th vertex.
    """
    d = 0
    for j_id in range(len(j_ptr)-1):
        # No self-loops
        if i_id == j_id:
            continue

        # Get non-zero in strengths for vertex out_row of label out_label
        r = j_ptr[j_id]
        s = j_ptr[j_id + 1]
        j_label = j_labels[r:s]
        j_val = j_vals[r:s]

        if r == s:
            continue

        # Get pij
        d += stripe_pij_alpha(p_f, param, i_label, i_val,
                              j_label, j_val)

    return d
    

@jit(nopython=True)
def stripe_eq_constr_alpha(x, p_f, s_out_i, s_out_j, s_out_w,
                           s_in_i, s_in_j, s_in_w, num_e):
    exp_e = 0
    for out_row in range(len(s_out_i)-1):
        # Get non-zero out strengths for vertex out_row of label out_label
        n = s_out_i[out_row]
        m = s_out_i[out_row + 1]

        if n == m:
            continue

        out_label = s_out_j[n:m]
        out_vals = s_out_w[n:m]
        exp_e += stripe_exp_degree_vertex(
            p_f, x, out_row, out_label, out_vals, s_in_i, s_in_j, s_in_w)

    return exp_e - num_e


@jit(nopython=True)
def stripe_eq_jac_alpha(x, p_f, jac_f, s_out_i, s_out_j, s_out_w,
                        s_in_i, s_in_j, s_in_w):
    """ Calculate derivative of expected edges in min degree model.
    """
    jac = np.zeros(2, dtype=np.float64)

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
            res = stripe_pij_jac_alpha(p_f, jac_f, x, out_label, out_vals,
                                       in_label, in_vals)

            jac[0] += res[0]
            jac[1] += res[1]

    return jac


@jit(nopython=True)
def stripe_ineq_constr_alpha(x, p_f, i_id, i_label, i_val, 
                             j_ptr, j_labels, j_vals):
    deg = stripe_exp_degree_vertex(
        p_f, x, i_id, i_label, i_val, j_ptr, j_labels, j_vals)
    return deg - 1


@jit(nopython=True)
def stripe_ineq_jac_alpha(x, p_f, jac_f, i_id, i_label, i_val,
                          j_ptr, j_labels, j_vals):
    jac = np.zeros(2, dtype=np.float64)
    for j_id in range(len(j_ptr)-1):
        # No self-loops
        if i_id == j_id:
            continue

        # Get non-zero in strengths for vertex out_row of label out_label
        r = j_ptr[j_id]
        s = j_ptr[j_id + 1]
        j_label = j_labels[r:s]
        j_val = j_vals[r:s]

        if r == s:
            continue

        res = stripe_pij_jac_alpha(p_f, jac_f, x,
                                   i_label, i_val,
                                   j_label, j_val)
        jac[0] += res[0]
        jac[1] += res[1]

    return jac


@jit(nopython=True)
def stripe_f_jac(p_f, jac_f, param, s_out_i, s_out_j, s_out_w,
                 s_in_i, s_in_j, s_in_w, n_edges):
    """ Compute the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    f = np.float64(0)
    jac = np.float64(0)

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
            res = stripe_pij_f_jac(
                p_f, jac_f, param, out_label, out_vals, in_label, in_vals)
            f += res[0]
            jac += res[1]

    return f - n_edges, jac


@jit(nopython=True)
def stripe_exp_edges(p_f, param, s_out_i, s_out_j, s_out_w,
                     s_in_i, s_in_j, s_in_w, per_label):
    """ Compute the expected number of edges.

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
    exp_edges = np.float64(0)

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
            pij = stripe_pij(
                p_f, param, out_label, out_vals, in_label, in_vals, per_label)
            exp_edges += pij

    return exp_edges


@jit(nopython=True)
def stripe_exp_edges_label(p_f, param, out_strength, in_strength, num_labels,
                           per_label):
    """ Compute the expected number of edges with one parameter controlling
    for the density for each label.
    """
    exp_edges = np.zeros(num_labels, dtype=np.float64)

    for i in range(num_labels):
        if per_label:
            x = param[:, i]
        else:
            x = param[:, 0]

        exp_edges[i] = layer_exp_edges(
            p_f,
            x,
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i])

    return exp_edges


@jit(nopython=True)
def stripe_exp_degree(p_f, param, s_out_i, s_out_j, s_out_w,
                      s_in_i, s_in_j, s_in_w, N, per_label):
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
    degree = np.zeros(N, dtype=np.float64)
    out_degree = np.zeros(N, dtype=np.float64)
    in_degree = np.zeros(N, dtype=np.float64)

    # Iterate over vertex ids of the out strength and compute for each id the
    # expected degree
    for i in np.arange(len(s_out_i) - 1):
        # Get non-zero out strengths for vertex i
        n = s_out_i[i]
        m = s_out_i[i + 1]

        out_l_i = s_out_j[n:m]
        out_v_i = s_out_w[n:m]

        # Get non-zero in strengths for vertex i
        n = s_in_i[i]
        m = s_in_i[i + 1]

        in_l_i = s_in_j[n:m]
        in_v_i = s_in_w[n:m]

        for j in np.arange(i + 1, len(s_in_i) - 1):
            # Get non-zero out strengths for vertex j
            n = s_out_i[j]
            m = s_out_i[j + 1]

            out_l_j = s_out_j[n:m]
            out_v_j = s_out_w[n:m]

            # Get non-zero in strengths for vertex j
            n = s_in_i[j]
            m = s_in_i[j + 1]

            in_l_j = s_in_j[n:m]
            in_v_j = s_in_w[n:m]

            # Get pij
            pij = stripe_pij(p_f, param, out_l_i, out_v_i,
                             in_l_j, in_v_j, per_label)
            pji = stripe_pij(p_f, param, out_l_j, out_v_j,
                             in_l_i, in_v_i, per_label)
            p = 1 - (1 - pij)*(1 - pji)
            degree[i] += p
            degree[j] += p
            out_degree[i] += pij
            out_degree[j] += pji
            in_degree[j] += pij
            in_degree[i] += pji

    return degree, out_degree, in_degree


@jit(nopython=True)
def stripe_exp_degree_label(p_f, param, out_strength, in_strength, num_labels,
                            per_label):
    """ Compute the expected degree by label for the stripe fitness model
    with one parameter controlling for the density for each label.
    """
    exp_d_out = []
    exp_d_in = []
    for i in range(num_labels):
        if per_label:
            x = param[:, i]
        else:
            x = param[:, 0]

        res = layer_exp_degree(
            p_f,
            x,
            out_strength[out_strength.label == i],
            in_strength[in_strength.label == i],
            i)
        exp_d_out.extend(res[0])
        exp_d_in.extend(res[1])

    return exp_d_out, exp_d_in


@jit(nopython=True)
def stripe_av_nn_prop(p_f, param, prop, ndir, s_out_i, s_out_j, s_out_w,
                      s_in_i, s_in_j, s_in_w, per_label):
    """ Compute the expected number of edges.
    """
    av_nn = np.zeros(prop.shape, dtype=np.float64)

    # Iterate over vertex ids
    for i in np.arange(len(s_out_i) - 1):
        # Get non-zero out strengths for vertex i
        n = s_out_i[i]
        m = s_out_i[i + 1]

        out_l_i = s_out_j[n:m]
        out_v_i = s_out_w[n:m]

        # Get non-zero in strengths for vertex i
        n = s_in_i[i]
        m = s_in_i[i + 1]

        in_l_i = s_in_j[n:m]
        in_v_i = s_in_w[n:m]

        for j in np.arange(i + 1, len(s_in_i) - 1):
            # Get non-zero out strengths for vertex j
            n = s_out_i[j]
            m = s_out_i[j + 1]

            out_l_j = s_out_j[n:m]
            out_v_j = s_out_w[n:m]

            # Get non-zero in strengths for vertex j
            n = s_in_i[j]
            m = s_in_i[j + 1]

            in_l_j = s_in_j[n:m]
            in_v_j = s_in_w[n:m]

            # Get pij
            pij = stripe_pij(p_f, param, out_l_i, out_v_i,
                             in_l_j, in_v_j, per_label)
            pji = stripe_pij(p_f, param, out_l_j, out_v_j,
                             in_l_i, in_v_i, per_label)

            if ndir == 'out':
                av_nn[i] += pij*prop[j]
                av_nn[j] += pji*prop[i]
            elif ndir == 'in':
                av_nn[i] += pji*prop[j]
                av_nn[j] += pij*prop[i]
            elif ndir == 'out-in':
                p = 1 - (1 - pij)*(1 - pji)
                av_nn[i] += p*prop[j]
                av_nn[j] += p*prop[i]
            else:
                raise ValueError('Direction of neighbourhood not right.')

    return av_nn


@jit(nopython=True)
def stripe_likelihood(adj_k, adj_i, adj_j, p_f, param, out_strength,
                      in_strength, num_labels, per_label, lgs):
    """ Compute the binary log likelihood of a graph given the fitted model.
    """
    if lgs:
        like = 0
    else:
        like = 1

    for k in range(num_labels):
        s_out = out_strength[out_strength.label == k]
        s_in = in_strength[in_strength.label == k]

        for i in np.arange(len(s_out)):
            out_i = s_out[i].id
            out_v = s_out[i].value

            # Get list of indices for ith vertex in kth layer
            o = adj_k[k]
            m = adj_i[k, out_i]
            n = adj_i[k, out_i+1]
            j_list = adj_j[o+m:o+n]
            for j in np.arange(len(s_in)):
                in_j = s_in[j].id
                in_v = s_in[j].value
                if out_i != in_j:
                    if per_label:
                        p = p_f(param[:, k], out_v, in_v)
                    else:
                        p = p_f(param[:, 0], out_v, in_v)
                else:
                    p = 0

                # Check if link exists
                if in_j in j_list:
                    if lgs:
                        like += log(p)
                    else:
                        like *= p
                else:
                    if lgs:
                        like += log(1 - p)
                    else:
                        like *= 1 - p
    if lgs:
        return like
    else:
        return log(like)


@jit(nopython=True)
def stripe_sample(p_f, param, out_strength, in_strength, num_labels,
                  per_label):
    """ Sample edges and weights from the stripe ensemble.
    """
    sample = []
    for i in range(num_labels):
        s_out = out_strength[out_strength.label == i]
        s_in = in_strength[in_strength.label == i]
        if per_label:
            sample.extend(
                layer_sample(p_f, param[:, i], s_out, s_in, i))
        else:
            sample.extend(
                layer_sample(p_f, param[:, 0], s_out, s_in, i))
        
    return np.array(sample, dtype=np.float64)


# --------------- BLOCK METHODS ---------------
@jit(nopython=True)
def block_exp_vertex_degree(p_f, param, out_g, out_vals, in_gj, in_vals):
    d = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                d += p_f(param, out_vals[i], in_vals[j])

    return d


@jit(nopython=True)
def block_f_jac_i(p_f, jac_f, param, out_g, out_vals, in_gj, in_vals):
    f = 0
    jac = 0
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                f += p_f(param, out_vals[i], in_vals[j])
                jac += jac_f(param, out_vals[i], in_vals[j])
    return f, jac


@jit(nopython=True)
def block_exp_num_edges(p_f, param, s_out_i, s_out_j, s_out_w,
                        s_in_i, s_in_j, s_in_w, group_arr):
    """ Calculate the expected number of edges for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    param: np.ndarray
        parameters vector
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
        num += block_exp_vertex_degree(
            p_f, param, out_g, out_vals, in_gj, in_vals)

    return num


@jit(nopython=True)
def block_exp_out_degree(p_f, param, s_out_i, s_out_j, s_out_w,
                         s_in_i, s_in_j, s_in_w, group_arr):
    """ Calculate the expected out degree for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    param: np.ndarray
        value of the density parameter
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
            p_f, param, out_g, out_vals, in_gj, in_vals)

    return out_degree


@jit(nopython=True)
def block_f_jac(p_f, jac_f, param, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j,
                s_in_w, group_arr, num_e):
    """ Calculate the objective function and its Jacobian for the block model.

    It is assumed that the arguments passed are the three arrays of two sparse
    matrices where the first index represents the vertex id, and the second a
    group. s_out must be a csr matrix and s_in a csc matrix.

    Arguments
    ----------
    param: np.ndarray
        value of the density parameter
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
    f = np.float64(0.0)
    jac = np.float64(0.0)

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
        res = block_f_jac_i(p_f, jac_f, param, out_g, out_vals, in_gj, in_vals)
        f += res[0]
        jac += res[1]

    return f - num_e, jac


@jit(nopython=True)
def block_iterative(param, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j, s_in_w,
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
                    aux += tmp / (1 + param[0]*tmp)

    return num_e/aux


@jit(nopython=True)
def block_sample_vertex(p_f, param, out_i, out_g, out_vals,
                        in_j, in_gj, in_vals, s_tot):
    """ Sample edges going out from a single vertex.
    """
    sample = []
    for i in range(len(out_g)):
        for j in range(len(in_gj)):
            if out_g[i] == in_gj[j]:
                p = p_f(param, out_vals[i], in_vals[j])
                if rng.random() < p:
                    w = np.float64(
                        rng.exponential(out_vals[i]*in_vals[j]/(s_tot*p)))
                    sample.append((out_i, in_j[j], w))

    return sample


@jit(nopython=True)
def block_sample(p_f, param, s_out_i, s_out_j, s_out_w,
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
            block_sample_vertex(p_f, param, out_row, out_g, out_vals,
                                in_j[notself], in_gj, in_vals, s_tot))

    return np.array(sample, dtype=np.float64)
