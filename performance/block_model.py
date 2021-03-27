""" Evaluate the performance of the block model in a typical pipeline.
"""
from time import perf_counter
import graph_ensembles as ge
import numpy as np


N = int(1e5)
L = 1e6
W = 9.364e9
G = np.random.randint(0, 99, shape=N)

start = perf_counter()
g_rand = ge.RandomGraph(num_vertices=N, num_edges=L, total_weight=W,
                        group_dict=G, discrete_weights=False)
g_rand.fit()
perf = perf_counter() - start
print('Time for random graph initialization and fit: ', perf)

start = perf_counter()
g = g_rand.sample()
perf = perf_counter() - start
print('Time for random graph sample: ', perf)

start = perf_counter()
block = ge.BlockFitnessModel(g)
perf = perf_counter() - start
print('Time for block init: ', perf)

start = perf_counter()
block.fit(method='newton')
perf = perf_counter() - start
print('Time for newton fit: ', perf)

print(block.solver_output.n_iter)

if not np.isclose(block.expected_num_edges(), block.num_edges,
                  atol=1e-8, rtol=0):
    print(block.expected_num_edges() - block.num_edges)

start = perf_counter()
block.fit(method='fixed-point')
perf = perf_counter() - start
print('Time for fixed-point fit: ', perf)

print(block.solver_output.n_iter)

if not np.isclose(block.expected_num_edges(), block.num_edges,
                  atol=1e-8, rtol=0):
    print(block.expected_num_edges() - block.num_edges)

start = perf_counter()
g_sample = block.sample()
perf = perf_counter() - start
print('Time for block sample: ', perf)
