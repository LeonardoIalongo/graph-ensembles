""" Test the performance of the stripe fitness model class on a set of graphs.
    The graphs can be generated using label_graph_gen.py in this folder.
"""
import os
import pickle as pk
import time
from time import perf_counter
import graph_ensembles as ge
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test performance of stripe model'
                                 ' per label on example graphs.')
parser.add_argument('--quick', action='store_true',
                    help='Perform only on smaller graph.')
parser.add_argument('-tol', default=1e-5, type=float)
parser.add_argument('-xtol', default=1e-6, type=float)

args = parser.parse_args()

test_names = ['init', 'newton', 'fixed-p', 'edges', 'e_lbl', 'degrees', 
              'd_lbl', 'k_nn', 's_nn', 'like', 'sample']
test_times = []
test_succ = []
graph_names = []

test_start = time.time()

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
    model = ge.StripeFitnessModel(g, per_label=True)
    perf = perf_counter() - start
    print('Time for model init: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(isinstance(model, ge.GraphEnsemble))

    print('Attempting newton fit:')
    start = perf_counter()
    model.fit(tol=args.tol, method='newton', verbose=True)
    perf = perf_counter() - start
    print('Time for newton fit: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

    if not np.allclose(model.expected_num_edges_label(get=True),
                       g.num_edges_label, atol=args.tol, rtol=0):
        print('Distance from root: ',
              model.exp_num_edges_label - g.num_edges_label)

    print('Attempting fixed-point fit:')
    start = perf_counter()
    model.fit(xtol=args.xtol, method='fixed-point', verbose=True)
    perf = perf_counter() - start
    print('Time for fixed-point fit: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(np.all([sol.converged for sol in model.solver_output]))

    if not np.allclose(model.expected_num_edges_label(get=True),
                       g.num_edges_label, atol=args.tol, rtol=0):
        print('Distance from root: ',
              model.exp_num_edges_label - g.num_edges_label)

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
    out_deg = model.exp_out_degree
    in_deg = model.exp_in_degree
    perf = perf_counter() - start
    print('Time for expected degrees: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

    start = perf_counter()
    out_deg = model.expected_out_degree_by_label(get=True)
    in_deg = model.exp_in_degree_label
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
    succ_tmp.append(np.all(meas >= 0))

    start = perf_counter()
    meas = model.expected_av_nn_strength(get=True)
    perf = perf_counter() - start
    print('Time for expected av_nn_strength: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(np.all(meas >= 0))
    
    start = perf_counter()
    meas = model.log_likelihood(g)
    perf = perf_counter() - start
    print('Time for log_likelihood: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(meas <= 0)

    start = perf_counter()
    g_sample = model.sample()
    perf = perf_counter() - start
    print('Time for model sample: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(isinstance(g_sample, ge.WeightedLabelGraph))

    test_times.append(times_tmp)
    test_succ.append(succ_tmp)

    if args.quick:
        break

time_format = time.strftime(
    '%H:%M:%S', time.gmtime(time.time() - test_start))
print('Total test time: ', time_format)

for i in range(len(graph_names)):
    print('\n--------------------------')
    print('Graph:', graph_names[i])
    print('Tests:', *test_names, sep='\t')
    print('Time:', *test_times[i], sep='\t')
    print('Status:', *test_succ[i], sep='\t')
