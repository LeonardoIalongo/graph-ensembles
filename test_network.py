""" This script tests the vector fitness reconstruction methodology on a 
simple network. """
import numpy as np 
import pandas as pd 
from ensembles import *

# Define network as edge and vertex list
edges = pd.DataFrame({
    'src': [2, 3, 1, 4, 3], 
    'dst': [1, 2, 4, 1, 1], 
    'weight': [5, 3, 2, 1, 3]}, dtype=np.int8)

edges = edges.astype({'weight':'float64'})

vertices = pd.DataFrame({
    'id': [1, 2, 3, 4], 
    'group': [1, 2, 3, 3]}, dtype=np.int8)


# Compute the strength sequence
strength = get_strenghts(edges, vertices, group_col = 'group')

print(strength)