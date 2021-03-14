""" Comparison of performance of different methods to fit the z parameters in
the stripe model with multiple z.
"""

from numba import jit
import numpy as np
import time
import matplotlib.pyplot as plt
import graph_ensembles as ge


# @jit(nopython=True)
def random_strenghts(n, l, g, w):
    """ Initialize random adjacency matrix for n nodes, l links, and g labels.
    """

    # Define indexes of non-zero elements of adjacency matrix
    nonzero_elems = np.zeros((l, 4), dtype=np.int64)
    nonzero_elems[:, 0:2] = np.random.randint(low=0, high=n, size=(l, 2))
    nonzero_elems[:, 2] = np.random.randint(low=0, high=g, size=l)
    nonzero_elems[:, 3] = np.random.randint(low=1, high=w, size=l)

    # Compute margins
    indexes = set()
    out_strength = np.zeros((n, g))
    in_strength = np.zeros((n, g))
    real_l = np.zeros(g)
    for link in nonzero_elems:
        # Ensure no-self loops
        if link[0] != link[1]:
            # Count number of non-zeros elements of the adj mat
            if (link[0], link[1], link[2]) not in indexes:
                indexes.add((link[0], link[1], link[2]))
                real_l[link[2]] += 1

            out_strength[link[0], link[2]] += link[3]
            in_strength[link[1], link[2]] += link[3]

    # Check that the in and out strengths match per group
    assert np.all(np.sum(out_strength, axis=0) == np.sum(in_strength, axis=0))

    return out_strength, in_strength


def einstein(s_out, s_in, z):
    """ Einstein notation method for computing the expected number of links.
    """
    g = s_out.shape[1]
    L = np.zeros(g)
    for k in np.arange(g):
        L[k] = np.einsum('', s_out[:, k], s_in[:, k])
    return L


# Specify parameters' space for graph construction
# Number of nodes
N = np.array([1e2, 1e3, 1e4, 1e5]).astype(int)

# Density of links as a fraction of max value N*(N-1)
L = (N * 0.1).astype(int)

# Number of groups as percentage of N
G = (N * 0.03).astype(int)

# Max strength for a node
w = 100

# Initialize computation time list
comp_time = np.zeros((len(N), 1))
i = 0
for n, l, g in zip(N, L, G):
    print(n, end=' ')

    # Generate random strengths
    s_out, s_in = random_strenghts(n, l, g, w)

    model = ge.StripeFitnessModel(s_out, s_in, l)
    model.z = 1

    # Solve and check that solution is admissible
    start = time.perf_counter()
    model.probability_array
    eval_time = time.perf_counter() - start
    comp_time[i, 0] = eval_time
    print('Done in: ', eval_time)
    i += 1

# Plot results, each model a series
for i in np.arange(comp_time.shape[1]):
    plt.loglog(N, comp_time[:, i],
               label=f"model={i}")
plt.legend(title='Parameters:')
plt.xlabel('Number of nodes')
plt.ylabel('Computation time (s)')
plt.show()
