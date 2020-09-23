""" This module defines the functions that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

import numpy as np
from scipy.optimize import fsolve
import scipy.sparse as sp


class GraphModel():
    """ General class for graph models. """

    def __init__(self, *args):
        pass


class VectorFitnessModel(GraphModel):
    """ A generalized fitness model that allows for vector strength sequences.

    Attributes
    ----------
    out_strength: np.ndarray or scipy.sparse matrix
        the out strength matrix
    in_strength: np.ndarray or scipy.sparse matrix
        the in strength matrix
    num_links: np.int
        the total number of links
    num_nodes: np.int
        the total number of nodes
    num_groups: np.int
        the total number of groups by which the vector strengths are computed
    """

    def __init__(self, out_strength, in_strength, num_links):
        """ Return a VectorFitnessModel for the given marginal graph data.

        The assumption is that the row number of the strength matrices
        represent the node number, while the column index relates to the
        group.

        Parameters
        ----------
        out_strength: np.ndarray or scipy.sparse matrix
            the out strength matrix of a graph
        in_strength: np.ndarray or scipy.sparse matrix
            the in strength matrix of a graph
        num_links: np.int
            the number of links in the graph
        param: np.ndarray
            array of parameters to be fitted by the model

        Returns
        -------
        VectorFitnessModel
            the model for the given input data
        """

        # Check that inputs are numpy arrays or scipy.sparse matrices
        if isinstance(out_strength, (np.ndarray, sp.spmatrix)):
            self.out_strength = out_strength
        else:
            raise TypeError('Out degree provided is neither a numpy array, '
                            'nor a scipy sparse matrix.')

        if isinstance(in_strength, (np.ndarray, sp.spmatrix)):
            self.in_strength = in_strength
        else:
            raise TypeError('Out degree provided is neither a numpy array, '
                            'nor a scipy sparse matrix.')

        if isinstance(num_links, int):
            self.num_links = num_links
        else:
            raise TypeError('Number of links not an integer.')

        # Check that dimensions are consistent
        msg = 'In and out strength do not have the same dimensions.'
        assert in_strength.shape == out_strength.shape, msg
        self.num_nodes = out_strength.shape[0]
        self.num_groups = out_strength.shape[1]

    def solve(self, z0=1):
        """ Fit parameters to match the ensemble to the provided data."""
        self.z = density_solver(lambda x: vector_fitness_link_prob(
                                    self.out_strength,
                                    self.in_strength,
                                    x,
                                    self.group_dict),
                                self.L,
                                z0)

    @property
    def probability_matrix(self):
        if hasattr(self, 'z'):
            return vector_fitness_link_prob(self.out_strength,
                                            self.in_strength,
                                            self.z,
                                            self.group_dict)
        else:
            print('Running solver before returning matrix.')
            self.solve()
            return vector_fitness_link_prob(self.out_strength,
                                            self.in_strength,
                                            self.z,
                                            self.group_dict)


def vector_fitness_link_prob(out_strength, in_strength, z):
    """Compute the link probability matrix given the in and out strength
    sequence, and the density parameter z.

    The out and in strength sequences should be numpy arrays or scipy.sparse
    matrices of one or two dimension. It is assumed that the index along the
    first dimension identifies the node, while the index along the second
    dimension relates to the grouping by which the strength is computed.

    Parameters
    ----------
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    z: np.float64
        the density parameter of the fitness model

    Returns
    -------
    numpy.ndarray
        the link probability matrix

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider avoiding computation of zeros and to return
    function or iterator.
    """
    # Check that dimensions are consistent
    msg = 'In and out strength do not have the same dimensions.'
    assert in_strength.shape == out_strength.shape, msg

    # Get number of nodes and groups
    N = out_strength.shape[0]
    G = out_strength.shape[1]

    # Initialize empty result
    p = np.zeros((N, N, G), dtype=np.float64)

    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(G):
                if i != j:
                    s_i = out_strength[i, k]
                    s_j = in_strength[j, k]
                    p[i, j, k] = z*s_i*s_j / (1 + z*s_i*s_j)

    return p


def density_solver(p_fun, L, z0):
    """ Return the optimal z to match a given number of links L.

    Note it is assumed that the probability matrix expresses the probability
    of extracting each edge as an independent Bernoulli trial. Such that the
    expected number of links in the network is the sum of the elements of the
    probability matrix.

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
    p_mat = p_fun(z0)
    if isinstance(p_mat, np.ndarray):
        return fsolve(lambda x: np.sum(p_fun(x)) - L, z0)
    elif isinstance(p_mat, sp.spmatrix):
        return fsolve(lambda x: p_fun(x).sum() - L, z0)


def from_pandas_edge_list(edges, vertices, group_col=None, group_dir='in'):
    """ TODO: transform in constructor for graph object from pandas edge list.

    Return the in and out strength sequences for the given network
    specified by an edge and vertex list as pandas dataframes.

    If a group_col is given then it returns a vector for each strength where
    each element is the strength related to each group. You can specify
    whether the grouping applies only to the 'in', 'out', or 'all' edges
    through group_dir. It also returns a dictionary that returns the group
    index give the node index and a another that given the identifier of the
    node, returns the index of it.
    """

    # Check that there are no duplicates in vertex definitions
    if any(vertices.loc[:, 'id'].duplicated()):
        raise ValueError('Duplicated node definitions.')

    # Check no duplicate edges
    if any(edges.loc[:, ['src', 'dst']].duplicated()):
        raise ValueError('There are duplicated edges.')

    # Construct dictionaries
    if group_col is None:
        i = 0
        index_dict = {}
        for index, row in vertices.iterrows():
            index_dict[row.id] = i
            i += 1
        N = len(vertices)

        out_temp = edges.groupby(['src']).agg({'weight': sum})
        out_strength = np.zeros(N)
        for index, row in out_temp.iterrows():
            out_strength[index_dict[index]] = row.weight

        in_temp = edges.groupby(['dst']).agg({'weight': sum})
        in_strength = np.zeros(N)
        for index, row in in_temp.iterrows():
            in_strength[index_dict[index]] = row.weight

        return out_strength, in_strength, index_dict

    else:
        i = 0
        j = 0
        index_dict = {}
        group_dict = {}
        group_list = vertices.loc[:, group_col].unique().tolist()
        for index, row in vertices.iterrows():
            index_dict[row.id] = i
            group_dict[i] = group_list.index(row[group_col])
            i += 1
            j += 1
        N = len(vertices)
        G = len(group_list)

        if group_dir in ['out', 'all']:
            out_strength = np.zeros((N, G))
            for index, row in edges.iterrows():
                i = index_dict[row.src]
                j = group_dict[index_dict[row.src]]
                out_strength[i, j] += row.weight
        else:
            out_temp = edges.groupby(['src']).agg({'weight': sum})
            out_strength = np.zeros(N)
            for index, row in out_temp.iterrows():
                out_strength[index_dict[index]] = row.weight

        if group_dir in ['in', 'all']:
            in_strength = np.zeros((N, G))
            for index, row in edges.iterrows():
                i = index_dict[row.dst]
                j = group_dict[index_dict[row.src]]
                in_strength[i, j] += row.weight
        else:
            in_temp = edges.groupby(['dst']).agg({'weight': sum})
            in_strength = np.zeros(N)
            for index, row in in_temp.iterrows():
                in_strength[index_dict[index]] = row.weight

        return out_strength, in_strength, index_dict, group_dict
