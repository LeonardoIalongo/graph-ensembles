""" Evaluate the performance of the stripe fitness model iteration step.

By iteration step we mean the objective function being minimised. In the case
of the stripe model it is the expected number of links for a given z.
It plots how the computation time evolves with increasing number of links and
edges. The results are platform dependent but the relative performance of
improvements to the models should be tested.

"""
import time
import graph_ensembles as ge
import numpy as np
import matplotlib.pyplot as plt

# Specify parameters' space for graph construction
# Number of nodes
N = [10, 50, 100, 500]

# Density of links as a fraction of max value N*(N-1)
L = [0.001, 0.01, 0.1]

# Number of groups as percentage of N
G = [0.1, 0.25, 0.5]

# Max strength for a node
W = [100]

# Initialize computation time list
comp_time = np.zeros((len(N), len(L), len(G), len(W)))
print('N, L, G, W')
for n, l, g, w in product(N, L, G, W):
    print(n, l, g, w, end=' ')
    # Define indexes of non-zero elements of adjacency matrix
    num_links = np.floor(l*n*(n-1)).astype(int)
    num_groups = np.floor(g*n).astype(int)
    adj = np.zeros((num_links, 4), dtype=np.int64)
    adj[:, 0:2] = np.random.randint(low=0, high=n,
                                    size=(num_links, 2), dtype=np.int64)
    adj[:, 2] = np.random.randint(low=0, high=num_groups,
                                  size=num_links, dtype=np.int64)
    adj[:, 3] = np.random.randint(low=1, high=w,
                                  size=num_links, dtype=np.int64)

    # Compute margins
    indexes = set()
    out_strength = np.zeros((n, num_groups))
    in_strength = np.zeros((n, num_groups))
    real_l = np.zeros(num_groups)
    for link in adj:
        # Ensure no-self loops
        if link[0] != link[1]:
            # Count number of non-zeros elements of the adj mat
            if (link[0], link[1], link[2]) not in indexes:
                indexes.add((link[0], link[1], link[2]))
                real_l[link[2]] += 1

            out_strength[link[0], link[2]] += link[3]
            in_strength[link[1], link[2]] += link[3]

    # Check that the in and out strengths match per group
    assert all(np.sum(out_strength, axis=0) == np.sum(in_strength, axis=0))

    # Create model
    model = ge.VectorFitnessModel(out_strength,
                                  in_strength,
                                  real_l)

    # Solve and check that solution is admissible
    start = time.perf_counter()
    model.z = np.ones(num_groups)
    prob_array = model.probability_array
    eval_time = time.perf_counter() - start
    comp_time[N.index(n), L.index(l), G.index(g), W.index(w)] = eval_time
    print('Done in: ', eval_time)

# Plot results grouping series by L, G, and W
for l, g, w in product(L, G, W):  # noqa: E741
    plt.loglog(N, comp_time[:, L.index(l), G.index(g), W.index(w)],
               label=f"L={l}, G={g}, W={w}")
plt.legend(title='Parameters:')
plt.xlabel('Number of nodes')
plt.ylabel('Computation time (s)')
plt.show()
