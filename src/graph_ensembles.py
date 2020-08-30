""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from scipy.optimize import fsolve
import scipy.sparse as sp


class Graph():
    """ Simple graph class """
    def __init__(self, *args):
        pass


class GraphModel():
    """ General class for graph models. """

    def __init__(self, *args):
        pass


class VectorFitnessModel(GraphModel):
    """ A generalized fitness model that allows vector strength sequences."""

    def __init__(self, *args):
        """ Return a VectorFitnessModel for the given marginal graph data.

        Accepts either a graph class object or three arguments (out_strength,
        in_strength, group_dict) as input.
        """
        num_args = len(args)
        if num_args < 1:
            self.out_strength = None
            self.in_strength = None
            self.group_dict = None

        elif num_args == 1:
            if not isinstance(args[0], Graph):
                raise ValueError('Only one argument was given but it is not a'
                                 ' graph.')
            # TODO: extract relevant data from graph
            pass

        elif num_args == 2:
            if any([isinstance(x, dict) for x in args]):
                raise ValueError('Missing one argument, probably either the'
                                 ' in or out strength sequence.')
            else:
                raise ValueError('Missing group dictionary.')

        elif num_args == 3:
            if isinstance(args[0], np.ndarray):
                self.out_strength = args[0]
            else:
                raise ValueError('Out degree provided is not a numpy array.')
            if isinstance(args[1], np.ndarray):
                self.in_strength = args[1]
            else:
                raise ValueError('In degree provided is not a numpy array.')
            if isinstance(args[2], dict):
                self.group_dict = args[2]
            else:
                raise ValueError('Group dict provided is not a dict.')

        else:
            raise ValueError('Too many arguments.')

    def solve(self):
        """ Fit parameters to match the ensemble to the provided data."""
        pass


def fitness_link_prob(out_strength, in_strength, z, N, group_dict=None):
    """Compute the link probability matrix given the in and out strength
    sequence, the density parameter z, and the number of vertices N.

    The out and in strength sequences should be numpy arrays of 1 dimension.
    If a group dictionary is specified then it will be assumed that the
    array will now be 2-dimensional and that the row relates to the node index
    while the column refers to the group. If there is only one dimension it is
    assumed that it is the total strength.

    Parameters
    ----------
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    z: np.float64
        the density parameter of the fitness model
    N: np.int64
        the number of vertices in the graph
    group_dict: dict
        a dictionary that given the index of a node returns its group

    Returns
    -------
    scipy.sparse.lil_matrix
        the link probability matrix

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider allowing for sparse matrices in case of groups and
    to avoid iteration over all indices.
    """

    # Initialize empty result
    p = sp.lil_matrix((N, N), dtype=np.float64)

    if group_dict is None:
        if (out_strength.ndim > 1) or (in_strength.ndim > 1):
            raise ValueError('A group dict was not provided but the strength '
                             + 'sequence is a vector.')
        else:
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i]
                        s_j = in_strength[j]
                        p[i, j] = z*s_i*s_j / (1 + z*s_i*s_j)
    else:
        if (out_strength.ndim > 1) and (in_strength.ndim > 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, group_dict[j]]
                        s_j = in_strength[j, group_dict[i]]
                        p[i, j] = z*s_i*s_j / (1 + z*s_i*s_j)
        elif (out_strength.ndim > 1) and (in_strength.ndim == 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i, group_dict[j]]
                        s_j = in_strength[j]
                        p[i, j] = z*s_i*s_j / (1 + z*s_i*s_j)
        elif (out_strength.ndim == 1) and (in_strength.ndim > 1):
            for i in np.arange(N):
                for j in np.arange(N):
                    if i != j:
                        s_i = out_strength[i]
                        s_j = in_strength[j, group_dict[i]]
                        p[i, j] = z*s_i*s_j / (1 + z*s_i*s_j)
        else:
            raise ValueError('A group dict was provided but no vector' +
                             ' strength sequence is available.')

    return p


def density_solver(p_fun, L, z0):
    """ Return the optimal z to match a given number of links L.

    Parameters
    ----------
    p_fun: function
        the function returning the probability matrix implied by a z value
    L : int
        number of links to be matched by expectation
    z0: np.float64
        initial conditions for z

    Returns
    -------
    np.float64
        the optimal z value solving L = <L>

    TODO: Currently implemented with general solver, consider iterative
    approach.
    """

    return fsolve(lambda x: np.sum(p_fun(x)) - L, z0)
