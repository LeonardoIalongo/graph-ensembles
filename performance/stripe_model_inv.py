
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

log = True
tol = 1e-5
xtol = 1e-6

test_names = ['init', 'fit', 'edges', 'e_lbl', 'degrees', 
              'd_lbl', 'k_nn', 's_nn', 'like', 'sample']
test_times = []
test_succ = []
graph_names = []

test_start = time.time()

with open("logs/stripe_inv.log", 'w') as f:

    if log:
        sys.stdout = f
        sys.stderr = f

    for filename in os.listdir('data/'):
        # Select .pk files
        if '.pk' not in filename:
            continue

        # Select multi layer files
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
        print('Number of labels: ', g.num_labels)

        # ------ Init and fit ------
        start = perf_counter()
        model = ge.StripeFitnessModel(g, scale_invariant=True)
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

        if not np.allclose(model.expected_num_edges_label(get=True),
                           g.num_edges_label, atol=tol, rtol=0):
            print('Distance from root: ',
                  model.expected_num_edges_label(get=True) - g.num_edges_label)

        # ------ Measures ------
        start = perf_counter()
        meas = model.expected_num_edges(get=True)
        perf = perf_counter() - start
        print('Time for expected edges: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(meas > 0)

        start = perf_counter()
        meas = model.expected_num_edges_label(get=True)
        perf = perf_counter() - start
        print('Time for expected edges by label: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(len(meas) == g.num_labels)
        
        start = perf_counter()
        deg = model.expected_degree(get=True)
        out_deg = model.expected_out_degree(get=True)
        in_deg = model.expected_in_degree(get=True)
        perf = perf_counter() - start
        print('Time for expected degrees: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

        start = perf_counter()
        out_deg = model.expected_out_degree_by_label(get=True)
        in_deg = model.expected_in_degree_by_label(get=True)
        perf = perf_counter() - start
        print('Time for expected degrees by label: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        res = True
        for i in range(g.num_labels):
            if not np.allclose(np.sum(out_deg[out_deg.label == i].value),
                               np.sum(in_deg[in_deg.label == i].value)):
                res = False
                break
        succ_tmp.append(res)

        start = perf_counter()
        meas = model.expected_av_nn_degree(get=True)
        perf = perf_counter() - start
        print('Time for expected av_nn_degree: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all(meas > 0))

        start = perf_counter()
        meas = model.expected_av_nn_strength(get=True)
        perf = perf_counter() - start
        print('Time for expected av_nn_strength: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(np.all(meas > 0))
        
        start = perf_counter()
        meas = model.log_likelihood(g)
        perf = perf_counter() - start
        print('Time for log_likelihood: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(meas < 0)

        start = perf_counter()
        g_sample = model.sample()
        perf = perf_counter() - start
        print('Time for model sample: ', perf)
        times_tmp.append('{:.3f}'.format(perf))
        succ_tmp.append(isinstance(g_sample, ge.WeightedLabelGraph))

    time_format = time.strftime(
        '%H:%M:%S', time.gmtime(time.time() - test_start))
    print('Total test time: ', time_format)

    for i in range(len(graph_names)):
        print('\n--------------------------')
        print('Graph:', graph_names[i])
        print('Tests:', *test_names, sep='\t')
        print('Time:', *test_times[i], sep='\t')
        print('Status:', *test_succ[i], sep='\t')
