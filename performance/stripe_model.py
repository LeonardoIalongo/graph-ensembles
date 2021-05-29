""" Test the performance of the stripe fitness model class on a set of graphs.
    The graphs can be generated using label_graph_gen.py in this folder.
"""
import os
import pickle as pk
import sys
import time
from time import perf_counter
import graph_ensembles as ge
import numpy as np

log = False
tol = 1e-5
xtol = 1e-6

test_names = ['init', 'newton', 'fixed-p', 'sample', 'degrees',
              'inv', 'inv-deg', 'min_deg', 'min_d-deg']
test_times = []
test_succ = []
graph_names = []

test_start = time.time()

with open("logs/stripe_fitness.log", 'w') as f:

    if log:
        sys.stdout = f
        sys.stderr = f

    for filename in os.listdir('data/'):
        # Select .pk files
        if '.pk' not in filename:
            continue

        # Select single layer files
        if '_l' not in filename:
            continue

        with open('data/' + filename, 'rb') as fl:
            g = pk.load(fl)

        graph_names.append(filename)
        times_tmp = []
        succ_tmp = []

        print('\n--------------------------')
        print('Testing on graph: ', filename)
        print('Number of vertices: ', g.num_vertices)
        print('Number of edges: ', g.num_edges)

        start = perf_counter()
        model = ge.StripeFitnessModel(g)
        perf = perf_counter() - start
        print('Time for model init: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(isinstance(model, ge.GraphEnsemble))

        print('Attempting newton fit:')
        start = perf_counter()
        model.fit(tol=tol, method='newton', verbose=True)
        perf = perf_counter() - start
        print('Time for newton fit: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

        if not np.allclose(model.expected_num_edges(), g.num_edges_label,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  model.expected_num_edges() - g.num_edges_label)

        print('Attempting fixed-point fit:')
        start = perf_counter()
        model.fit(xtol=1e-6, method='fixed-point', verbose=True)
        perf = perf_counter() - start
        print('Time for fixed-point fit: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

        if not np.allclose(model.expected_num_edges(), g.num_edges_label,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  model.expected_num_edges() - g.num_edges_label)

        start = perf_counter()
        g_sample = model.sample()
        perf = perf_counter() - start
        print('Time for model sample: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(isinstance(g_sample, ge.WeightedLabelGraph))

        start = perf_counter()
        out_deg = model.expected_out_degree()
        in_deg = model.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

        print('Attempting scale_invariant fit:')
        inv = ge.StripeFitnessModel(g, scale_invariant=True)
        start = perf_counter()
        inv.fit(tol=tol, verbose=True)
        perf = perf_counter() - start
        print('Time for invariant fit: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

        if not np.allclose(inv.expected_num_edges(), g.num_edges_label,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  inv.expected_num_edges() - g.num_edges_label)

        start = perf_counter()
        out_deg = inv.expected_out_degree()
        in_deg = inv.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

        print('Attempting min_degree fit:')
        a_model = ge.StripeFitnessModel(g, min_degree=True)
        start = perf_counter()
        a_model.fit(tol=tol, max_iter=1000, verbose=True)
        perf = perf_counter() - start
        print('Time for min_degree fit: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

        if not np.allclose(a_model.expected_num_edges(), g.num_edges_label,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  a_model.expected_num_edges() - g.num_edges_label)

        start = perf_counter()
        out_deg = a_model.expected_out_degree()
        in_deg = a_model.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

        test_times.append(times_tmp)
        test_succ.append(succ_tmp)

    time_format = time.strftime(
        '%H:%M:%S', time.gmtime(time.time() - test_start))
    print('Total test time: ', time_format)

    for i in range(len(graph_names)):
        print('\n--------------------------')
        print('Graph:', graph_names[i])
        print('Tests:', *test_names, sep='\t')
        print('Time:', *test_times[i], sep='\t')
        print('Status:', *test_succ[i], sep='\t')
