""" Test the performance of the fitness model class on a set of graphs.
    The graphs can be generated using graph_gen.py in this folder.
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

test_names = ['init', 'newton', 'fixed-p', 'edges', 'degrees',
              'k_nn', 's_nn', 'like', 'sample']
test_times = []
test_succ = []
graph_names = []

test_start = time.time()

for filename in os.listdir('data/'):
    # Select .pk files
    if '.pk' not in filename:
        continue

    # Select single layer files
    if '_z' not in filename:
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

    # ------ Init and fit ------
    start = perf_counter()
    model = ge.FitnessModel(g)
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
    succ_tmp.append(model.solver_output.converged)

    if not np.allclose(model.expected_num_edges(get=True), g.num_edges,
                       atol=args.tol, rtol=0):
        print('Distance from root: ',
              model.expected_num_edges(get=True) - g.num_edges)

    print('Attempting fixed-point fit:')
    start = perf_counter()
    model.fit(xtol=args.xtol, method='fixed-point', verbose=True)
    perf = perf_counter() - start
    print('Time for fixed-point fit: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(model.solver_output.converged)

    if not np.allclose(model.expected_num_edges(get=True), g.num_edges,
                       atol=args.tol, rtol=0):
        print('Distance from root: ',
              model.expected_num_edges(get=True) - g.num_edges)

    # ------ Measures ------
    start = perf_counter()
    meas = model.expected_num_edges(get=True)
    perf = perf_counter() - start
    print('Time for expected edges: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(meas > 0)

    start = perf_counter()
    deg = model.expected_degree(get=True)
    out_deg = model.exp_out_degree
    in_deg = model.exp_in_degree
    perf = perf_counter() - start
    print('Time for expected degrees: ', perf)
    times_tmp.append('{:.3f}'.format(perf))
    succ_tmp.append(np.allclose(np.sum(out_deg), np.sum(in_deg)))

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
    succ_tmp.append(isinstance(g_sample, ge.WeightedGraph))

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
