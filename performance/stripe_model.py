""" Evaluate the performance of the stripe model in a typical pipeline.
"""
from time import perf_counter
import graph_ensembles as ge
import numpy as np

N = int(1e4)
L = np.array([10, 35, 89, 3], dtype=np.uint64)
W = np.array([98, 25342, 1543, 532], dtype=np.float64)

start = perf_counter()
g_rand = ge.RandomGraph(num_vertices=N, num_edges=L, total_weight=W,
                        discrete_weights=True)
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
