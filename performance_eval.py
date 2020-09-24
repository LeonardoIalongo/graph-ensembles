""" This script is used to evaluate the performance of the network models.

It plots how the computation time evolves with increasing number of links and
edges. The results are platform dependent but the relative performance of
improvements to the models should be tested.
"""

import graph_ensembles as ge
import numpy as np

# Number of nodes series
N = np.logspace(1, 1, num=1, dtype=np.int64)

# Corresponding number of edges
L = N * 10

# Corresponding number of groups
G = np.floor(N / 3).astype('int')

# Corresponding max strength
W = N * 2

for n, l, g, w in zip(N, L, G, W):
    # Define non-zero elements of graph marginals
    out_strength = np.random.randint(low=0, high=2, size=(n, g),
                                     dtype=np.int64)
    in_strength = np.random.randint(low=0, high=2, size=(n, g),
                                    dtype=np.int64)

    # Add fitness value
    out_strength *= np.random.randint(low=1, high=w+1, size=(n, g),
                                      dtype=np.int64)
    in_strength *= np.random.randint(low=1, high=w+1, size=(n, g),
                                     dtype=np.int64)

    # Correct difference between total in and out strength per group
    diff = np.sum(out_strength, axis=0) - np.sum(in_strength, axis=0)

    # Distribute difference over all non zero elements
    for i in range(0, g):
        x = in_strength[:, i]
        x[x != 0] += np.floor(diff[i] / len(x[x != 0])).astype('int')
        x[x < 0] = 0

    # Give remainder to first element
    diff = np.sum(out_strength, axis=0) - np.sum(in_strength, axis=0)
    for i in range(0, g):
        row_mask = np.argmax(in_strength[:, i] > diff[i])
        in_strength[row_mask, i] += diff[i]

    assert all(np.sum(out_strength, axis=0) == np.sum(in_strength, axis=0))

    # Create model
    model = ge.VectorFitnessModel(out_strength,
                                  in_strength,
                                  l)

    # Solve and check that solution is admissible
    exp_num_links = model.probability_matrix.sum()
    print(l)
    print(exp_num_links)
