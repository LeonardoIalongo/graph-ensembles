""" Evaluate the performance of the block model in a typical pipeline.
"""
from time import perf_counter
import graph_ensembles as ge
import numpy as np


N = int(1e4)
L = 1e5
W = 9.364e9
G = np.random.randint(0, 99, N)

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

# Add groups
ge.lib.add_groups(g, G)

start = perf_counter()
block = ge.BlockFitnessModel(g)
perf = perf_counter() - start
print('Time for block init: ', perf)

start = perf_counter()
block.fit(method='newton')
perf = perf_counter() - start
print('Time for newton fit: ', perf)

print('Number of iterations: ', block.solver_output.n_iter)

if not np.isclose(block.expected_num_edges(), block.num_edges,
                  atol=1e-5, rtol=0):
    print('Distance from root: ', block.expected_num_edges() - block.num_edges)

start = perf_counter()
block.fit(method='fixed-point')
perf = perf_counter() - start
print('Time for fixed-point fit: ', perf)

print('Number of iterations: ', block.solver_output.n_iter)

if not np.isclose(block.expected_num_edges(), block.num_edges,
                  atol=1e-5, rtol=0):
    print('Distance from root: ', block.expected_num_edges() - block.num_edges)

start = perf_counter()
g_sample = block.sample()
perf = perf_counter() - start
print('Time for block sample: ', perf)

start = perf_counter()
out_deg = block.expected_out_degree()
in_deg = block.expected_in_degree()
perf = perf_counter() - start
print('Time for block expected degrees: ', perf)

inv = ge.BlockFitnessModel(g, scale_invariant=True)
start = perf_counter()
inv.fit()
perf = perf_counter() - start
print('Time for invariant fit: ', perf)

print('Number of iterations: ', inv.solver_output.n_iter)

if not np.isclose(inv.expected_num_edges(), inv.num_edges,
                  atol=1e-5, rtol=0):
    print('Distance from root: ',
          inv.expected_num_edges() - inv.num_edges)
