""" This script is used to evaluate the performance of the network models.

It plots how the computation time evolves with increasing number of links and
edges. The results are platform dependent but the relative performance of
improvements to the models should be tested.
"""
import time
import graph_ensembles as ge
import numpy as np

# Number of nodes series
N = np.logspace(1, 1, num=1, dtype=np.int64)

# Corresponding number of edges
L = N * 2

# Corresponding number of groups
G = np.floor(N / 3).astype('int')

# Corresponding max strength
W = N * 2

# Initialize computation time list
comp_time = []

for n, l, g, w in zip(N, L, G, W):
    # Define indexes of non-zero elements of adjacency matrix
    adj = np.zeros((l, 4), dtype=np.int64)
    adj[:, 0:2] = np.random.randint(low=0, high=n, size=(l, 2), dtype=np.int64)
    adj[:, 2] = np.random.randint(low=0, high=g, size=l, dtype=np.int64)
    adj[:, 3] = np.random.randint(low=1, high=w, size=l, dtype=np.int64)

    # Compute margins
    indexes = set()
    out_strength = np.zeros((n, g))
    in_strength = np.zeros((n, g))
    for link in adj:
        if (link[0], link[1]) not in indexes:
            indexes.add((link[0], link[1]))

        out_strength[link[0], link[2]] += link[3]
        in_strength[link[1], link[2]] += link[3]

    real_l = len(indexes)

    # Check that the in and out strengths match per group
    assert all(np.sum(out_strength, axis=0) == np.sum(in_strength, axis=0))

    # Create model
    model = ge.VectorFitnessModel(out_strength,
                                  in_strength,
                                  real_l)

    # Solve and check that solution is admissible
    start = time.perf_counter()
    model.solve()
    exp_num_links = model.probability_matrix.sum()
    comp_time.append(time.perf_counter() - start)
    np.testing.assert_almost_equal(real_l, exp_num_links, decimal=5)

print(comp_time)
