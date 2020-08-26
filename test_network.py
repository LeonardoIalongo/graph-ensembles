""" This script tests the vector fitness reconstruction methodology on a 
simple network. """
import numpy as np 
import pandas as pd 
from ensembles import *
import time

# Define network as edge and vertex list
edges = pd.DataFrame({
    'src': [2, 3, 1, 4, 3], 
    'dst': [1, 2, 4, 1, 1], 
    'weight': [5, 3, 2, 1, 3]}, dtype=np.int8)

edges = edges.astype({'weight':'float64'})

vertices = pd.DataFrame({
    'id': [1, 2, 3, 4], 
    'group': [1, 2, 3, 3]}, dtype=np.int8)

# Check no duplicate edges
if any(edges.loc[:, ['src', 'dst']].duplicated()):
    raise ValueError('Duplicated edges')

num_vertices = vertices.id.nunique()
num_edges = len(edges)

# Compute the strength sequence
out_strength = np.array([[2], [5], [6], [1]])
in_strength = np.array([[0, 5, 4], 
                         [0, 0, 3], 
                         [0, 0, 0], 
                         [2, 0, 0]])

# Compute the probability matrix given a z
z = 1.0

t0 = time.process_time()
p = fitness_link_prob(out_strength, in_strength[:,0], z, num_vertices)
print(time.process_time() - t0)

print(p.toarray())