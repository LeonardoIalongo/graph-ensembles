import numpy as np
import numpy.random as rng
from numba import jit
from math import isinf
from numpy.lib.recfunctions import merge_arrays
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
def sample_edges(z, out_strength, in_strength):
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
                    sample.append((ind_out, ind_in, w))

    return sample


def sample(z, out_strength, in_strength, num_vertices):
    # Generate uninitialised graph object
    g = graphs.WeightedGraph.__new__(graphs.WeightedGraph)

    # Initialise common object attributes
    g.num_vertices = num_vertices
    g.id_dtype = 'u' + str(mt.get_num_bytes(num_vertices))
    g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
        type=np.recarray, dtype=[('id', g.id_dtype)])
    g.id_dict = {}
    for x in g.v.id:
        g.id_dict[x] = x

    # Sample edges and extract properties
    e = np.array(sample_edges(z, out_strength, in_strength),
                 dtype=[('src', 'f8'),
                        ('dst', 'f8'),
                        ('weight', 'f8')]).view(type=np.recarray)

    e = e.astype([('src', g.id_dtype),
                  ('dst', g.id_dtype),
                  ('weight', 'f8')])
    g.sort_ind = np.argsort(e)
    g.e = e[g.sort_ind]
    g.num_edges = len(g.e)
    g.total_weight = np.sum(e.weight)

    return g


# Define random generation parameters
n_vertices_list = [10000, 10000, 100000]
gamma = 2
z_list = [2e-5, 7e-3, 1e-6]

for n_vertices, z in zip(n_vertices_list, z_list):
    # Sample node fitness
    s_out = merge_arrays((np.arange(n_vertices, dtype='u4'),
                         power_law(n_vertices, gamma)))
    s_out = s_out.astype(
        [('id', 'u4'), ('value', 'f8')]).view(type=np.recarray)

    s_in = merge_arrays((np.arange(n_vertices, dtype='u4'),
                        power_law(n_vertices, gamma)))
    s_in = s_in.astype([('id', 'u4'), ('value', 'f8')]).view(type=np.recarray)

    # # Visual inspection
    # import matplotlib.pyplot as plt
    # bins = np.logspace(0, 6, 100)
    # plt.hist(s_out.value, bins=bins, label='out')
    # plt.hist(s_in.value, bins=bins, label='in')
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # Compute expected edges
    exp_n = fit_exp_edges(z, s_out, s_in)
    print('Expected number of edges: ', exp_n)

    # Sample graph
    g = sample(z, s_out, s_in, n_vertices)

    print('Number of edges in sample: ', g.num_edges)

    # Save sample
    file_path = 'data/g_v{0:.1e}_e{1:.1e}_z{2:.1e}.pk'.format(
        n_vertices, exp_n, z).replace('+0', '').replace('-0', '')
    with open(file_path, 'wb') as f:
        pk.dump(g, f)
