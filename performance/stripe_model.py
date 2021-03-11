""" Evaluate the performance of the stripe model in a typical pipeline.
"""
from time import perf_counter
import graph_ensembles as ge
import numpy as np


N = int(1e3)
L = np.array([9.99e5, 3500, 890, 12], dtype=np.uint64)
W = np.array([9.3e6, 25342, 1543, 532], dtype=np.float64)

start = perf_counter()
g_rand = ge.RandomGraph(num_vertices=N, num_edges=L, total_weight=W,
                        discrete_weights=False)
g_rand.fit()
perf = perf_counter() - start
print('Time for random graph initialization and fit: ', perf)

start = perf_counter()
g = g_rand.sample()
perf = perf_counter() - start
print('Time for random graph sample: ', perf)

start = perf_counter()
stripe = ge.StripeFitnessModel(g)
perf = perf_counter() - start
print('Time for stripe init: ', perf)

start = perf_counter()
stripe.fit(method='newton')
perf = perf_counter() - start
print('Time for newton fit: ', perf)

if not np.allclose(stripe.expected_num_edges(), stripe.num_edges, atol=1e-8):
    print(stripe.expected_num_edges() - stripe.num_edges)

start = perf_counter()
stripe.fit(method='fixed-point')
perf = perf_counter() - start
print('Time for fixed-point fit: ', perf)

print(stripe.expected_num_edges() - stripe.num_edges)

if not np.allclose(stripe.expected_num_edges(), stripe.num_edges, atol=1e-8):
    print(stripe.expected_num_edges() - stripe.num_edges)
