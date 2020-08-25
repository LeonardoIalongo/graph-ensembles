""" This script tests the vector fitness reconstruction methodology on a 
simple network. """
import numpy as np 
import pandas as pd 

# Define network as pandas edge list
edge_list = pd.DataFrame({'src': [2, 3, 1, 4, 3], 
    'dst': [1, 2, 4, 1, 1], 'weight': [5, 3, 2, 1, 3]})


print(edge_list)