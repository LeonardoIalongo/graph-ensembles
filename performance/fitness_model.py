""" Test the performance of the fitness model class on a set of graphs.
    The graphs can be generated using graph_gen.py in this folder.
"""
import os
import pickle as pk
import sys
from time import perf_counter
import graph_ensembles as ge
import numpy as np

log = True
tol = 1e-5
xtol = 1e-6

with open("logs/fitness.log", 'w') as f:

    if log:
        sys.stdout = f
        sys.stderr = f

    for filename in os.listdir('data/'):
        # Select .pk files
        if '.pk' not in filename:
            continue

        # Select single layer files
        if '_z' not in filename:
            continue

        with open('data/' + filename, 'rb') as f:
            g = pk.load(f)

        print('\n--------------------------')
        print('Testing on graph: ', filename)
        print('Number of vertices: ', g.num_vertices)
        print('Number of edges: ', g.num_edges)

        start = perf_counter()
        model = ge.FitnessModel(g)
        perf = perf_counter() - start
        print('Time for model init: ', perf)

        print('Attempting newton fit:')
        start = perf_counter()
        model.fit(tol=tol, method='newton', verbose=True)
        perf = perf_counter() - start
        print('Time for newton fit: ', perf)

        if not np.allclose(model.expected_num_edges(), g.num_edges,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  model.expected_num_edges() - g.num_edges)

        print('Attempting fixed-point fit:')
        start = perf_counter()
        model.fit(xtol=1e-6, method='fixed-point', verbose=True)
        perf = perf_counter() - start
        print('Time for fixed-point fit: ', perf)

        if not np.allclose(model.expected_num_edges(), g.num_edges,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  model.expected_num_edges() - g.num_edges)

        start = perf_counter()
        g_sample = model.sample()
        perf = perf_counter() - start
        print('Time for model sample: ', perf)

        start = perf_counter()
        out_deg = model.expected_out_degree()
        in_deg = model.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)

        print('Attempting scale_invariant fit:')
        inv = ge.FitnessModel(g, scale_invariant=True)
        start = perf_counter()
        inv.fit(tol=tol, verbose=True)
        perf = perf_counter() - start
        print('Time for invariant fit: ', perf)

        if not np.allclose(inv.expected_num_edges(), g.num_edges,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  inv.expected_num_edges() - g.num_edges)

        start = perf_counter()
        out_deg = inv.expected_out_degree()
        in_deg = inv.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)

        print('Attempting min_degree fit:')
        a_model = ge.FitnessModel(g, min_degree=True)
        start = perf_counter()
        a_model.fit(tol=tol, verbose=True)
        perf = perf_counter() - start
        print('Time for min_degree fit: ', perf)

        if not np.allclose(a_model.expected_num_edges(), g.num_edges,
                           atol=tol, rtol=0):
            print('Distance from root: ',
                  a_model.expected_num_edges() - g.num_edges)

        start = perf_counter()
        out_deg = a_model.expected_out_degree()
        in_deg = a_model.expected_in_degree()
        perf = perf_counter() - start
        print('Time for model expected degrees: ', perf)
