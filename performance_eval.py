""" This script is used to evaluate the performance of the network models.

It plots how the computation time evolves with increasing number of links and
edges. The results are platform dependent but the relative performance of
improvements to the models should be tested.
"""

import graph_ensembles as ge
import numpy as np

# Number of links series
N = np.logspace(1, 5, num=5)

# Number of edges of corresponding series
L = N * 10

