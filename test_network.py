""" This script tests the vector fitness reconstruction methodology on a
simple network. """
import numpy as np
import pandas as pd
import ensembles as ge
import time

# Define network as edge and vertex list
edges = pd.DataFrame({
    'src': [2, 3, 1, 4, 3],
    'dst': [1, 2, 4, 1, 1],
    'weight': [5, 3, 2, 1, 3]}, dtype=np.int8)

edges = edges.astype({'weight': 'float64'})

vertices = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'group': [1, 2, 3, 3]}, dtype=np.int8)


num_vertices = vertices.id.nunique()
num_edges = len(edges)

# Compute the strength sequence
out_strength, in_strength, index_dict, group_dict = ge.get_strenghts(
    edges, vertices, group_col='group')

# Find correct z to replicate graph density
z0 = 1.0

t0 = time.process_time()
z = ge.density_solver(
    lambda x: ge.fitness_link_prob(out_strength, in_strength,
                                   x, num_vertices, group_dict),
    num_edges,
    z0)
print('Compute time:', time.process_time() - t0, 's')

P = ge.fitness_link_prob(out_strength, in_strength,
                         z, num_vertices, group_dict)

print(P.sum())
print(P.toarray())
