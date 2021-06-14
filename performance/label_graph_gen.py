import numpy as np
import numpy.random as rng
from numba import jit
from math import isinf
from graph_ensembles import graphs
from graph_ensembles import methods as mt
import pickle as pk


@jit(nopython=True)
def p_fit(d, x_i, x_j):
    tmp = d*x_i*x_j
    if isinf(tmp):
        return 1
    else:
        return tmp / (1 + tmp)


@jit(nopython=True)
def power_law(n, gamma, min_val=1):
    res = np.empty(n, dtype=np.float64)

    for i in range(n):
        res[i] = min_val * (1 - rng.random()) ** (-1 / (gamma - 1))

    return res


@jit(nopython=True)
def fit_exp_edges(z, fit_out, fit_in):
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
                exp_edges += p_fit(z, v_out, v_in)

    return exp_edges


@jit(nopython=True)
def sample_edges(z, out_strength, in_strength, label):
    s_tot = max(np.sum(out_strength.value), np.sum(in_strength.value))
    sample = []
    for i in np.arange(len(out_strength)):
        ind_out = out_strength[i].id
        s_out = out_strength[i].value
        for j in np.arange(len(in_strength)):
            ind_in = in_strength[j].id
            s_in = in_strength[j].value
            if ind_out != ind_in:
                p = p_fit(z, s_out, s_in)
                if rng.random() < p:
                    w = np.float64(rng.exponential(s_out*s_in/(s_tot*p)))
                    sample.append((label, ind_out, ind_in, w))

    return sample


def sample(z, out_strength, in_strength, num_vertices, num_labels):
    # Generate uninitialised graph object
    g = graphs.WeightedLabelGraph.__new__(graphs.WeightedLabelGraph)
    g.lv = graphs.LabelVertexList()

    # Initialise common object attributes
    g.num_vertices = num_vertices
    g.id_dtype = 'u' + str(mt.get_num_bytes(num_vertices))
    g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
        type=np.recarray, dtype=[('id', g.id_dtype)])
    g.id_dict = {}
    for x in g.v.id:
        g.id_dict[x] = x

    # Sample edges and extract properties
    e = []
    for label in range(num_labels):
        e.extend(sample_edges(z[label],
                              out_strength[out_strength.label == label],
                              in_strength[in_strength.label == label],
                              label))

    e = np.array(e)
    e = e.view(type=np.recarray,
               dtype=[('label', 'f8'),
                      ('src', 'f8'),
                      ('dst', 'f8'),
                      ('weight', 'f8')]).reshape((e.shape[0],))
    g.num_labels = num_labels
    g.label_dtype = 'u' + str(mt.get_num_bytes(num_labels))
    e = e.astype([('label', g.label_dtype),
                  ('src', g.id_dtype),
                  ('dst', g.id_dtype),
                  ('weight', 'f8')])
    g.sort_ind = np.argsort(e)
    g.e = e[g.sort_ind]
    g.num_edges = mt.compute_num_edges(g.e)
    ne_label = mt.compute_num_edges_by_label(g.e, g.num_labels)
    dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
    g.num_edges_label = ne_label.astype(dtype)
    g.total_weight = np.sum(e.weight)
    g.total_weight_label = mt.compute_tot_weight_by_label(
            g.e, g.num_labels)

    return g


def strengths(k, n, n_l):
    s_out = np.recarray(k, dtype=[('label', np.uint8),
                                  ('id', np.uint32),
                                  ('value', np.float64)
                                  ])
    s_in = np.recarray(k, dtype=[('label', np.uint8),
                                 ('id', np.uint32),
                                 ('value', np.float64)
                                 ])

    s_out.label = rng.randint(0, n_l, k)
    s_in.label = s_out.label.copy()
    s_out.id = rng.randint(0, n, k)
    s_in.id = rng.randint(0, n, k)
    s_out.value = power_law(k, 2)  # gamma = 2
    s_in.value = s_out.value.copy()

    s_out.sort()
    s_in.sort()

    return s_out, s_in


# Define random generation parameters
n_vertices_list = [10000, 10000, 100000]
n_labels = 100
z_list = [2e-4, 1e-3, 5e-6]

for n_vertices, z, n in zip(n_vertices_list, z_list, range(3)):
    # Sample node fitness
    if n == 0:
        s_out, s_in = strengths(n_vertices*5, n_vertices, n_labels)
    if n == 1:
        s_out, s_in = strengths(n_vertices*20, n_vertices, n_labels)
    else:
        s_out, s_in = strengths(n_vertices*10, n_vertices, n_labels)

    # # Visual inspection
    # import matplotlib.pyplot as plt
    # bins = np.logspace(0, 6, 100)
    # plt.hist(s_out.value, bins=bins, label='out')
    # plt.hist(s_in.value, bins=bins, label='in')
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # Add noise to z
    z = z*(2*rng.random(n_labels))**2

    # Compute expected edges
    exp_n = []
    for i in range(n_labels):
        exp_n.append(fit_exp_edges(z[i],
                                   s_out[s_out.label == i],
                                   s_in[s_in.label == i]))
    print('Expected number of edges: ',
          np.mean(exp_n), np.min(exp_n), np.max(exp_n))

    # Sample graph
    g = sample(z, s_out, s_in, n_vertices, n_labels)

    print('Number of edges in sample: ', g.num_edges)

    # Save sample
    file_path = 'data/g_v{0:.1e}_e{1:.1e}_l{2:.1e}.pk'.format(
        n_vertices, np.mean(exp_n), n_labels).replace(
        '+0', '').replace('-0', '')
    with open(file_path, 'wb') as f:
        pk.dump(g, f)
