""" Evaluate the performance of the stripe model in a typical pipeline.
"""
import time
import graph_ensembles as ge
import numpy as np

N = int(1e4)
L = np.array([10, 35, 89, 3], dtype=np.uint64)
# L = 100

start = time.perf_counter()
g_rand = ge.RandomGraph(num_vertices=N, num_edges=L)
g_rand.fit()
perf = time.perf_counter() - start
print('Time for random graph initialization and fit: ', perf)

start = time.perf_counter()
g = g_rand.sample()
perf = time.perf_counter() - start
print('Time for random graph sample: ', perf)

print(g.num_edges_label)
