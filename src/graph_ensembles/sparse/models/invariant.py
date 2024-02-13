from .fitness import FitnessModel
from .fitness import MultiFitnessModel
import numpy as np
from numba import jit
from math import isinf
from math import log
from math import expm1
from math import exp


class ScaleInvariantModel(FitnessModel):
    """The Scale Invariant model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    prop_out: np.ndarray
        The out fitness sequence.
    prop_in: np.ndarray
        the in fitness sequence.
    prop_dyad: function
        A function that returns the dyadic properties of two nodes.
    num_edges: int
        The total number of edges.
    num_vertices: int
        The total number of nodes.
    param: float
        The free parameters of the model.
    selfloops: bool
        Selects if self loops (connections from i to i) are allowed.

    Methods
    -------
    fit:
        Fit the parameters of the model with the given method.
    """

    def __init__(self, *args, **kwargs):
        """Return a ScaleInvariantModel for the given graph data.

        The model accepts as arguments either: a DiGraph, in which case the
        strengths are used as fitnesses, or directly the fitness sequences (in
        and out). The model accepts the fitness sequences as numpy arrays.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, x_i, y_j, z_ij):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        if (x_i == 0) or (y_j == 0) or (z_ij == 0):
            return 0.0, 0.0

        if d[0] == 0:
            return 0.0, x_i * y_j * z_ij

        tmp = x_i * y_j * z_ij
        tmp1 = d[0] * tmp
        if isinf(tmp1):
            return 1.0, 0.0
        else:
            return -expm1(-tmp1), tmp * exp(-tmp1)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (z_ij == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j * z_ij
        if isinf(tmp):
            return 1.0
        else:
            return -expm1(-tmp)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, x_i, y_j, z_ij):
        """Compute the log probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (z_ij == 0) or (d[0] == 0):
            return -np.infty

        tmp = d[0] * x_i * y_j * z_ij
        if isinf(tmp):
            return 0.0
        else:
            return log(-expm1(-tmp))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(d, x_i, y_j, z_ij):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        if (x_i == 0) or (y_j == 0) or (z_ij == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j * z_ij
        if isinf(tmp):
            return -np.infty
        else:
            return -tmp


class MultiInvariantModel(MultiFitnessModel):
    """A generalized Scale Invariant model that allows for fitnesses by label.

    This model allows to take into account labels of the edges and include
    this information as part of the model. Two quantities can be preserved by
    the ensemble: either the total number of edges, or the number of edges per
    label.

    Attributes
    ----------
    prop_out: list
        The out fitness by label as a list of tuples containing for each node
        the values of the non-zero elements and the relative label indices.
    prop_in: list
        The in fitness by label as a list of tuples containing for each node
        the values of the non-zero elements and the relative label indices.
    prop_dyad: function
        A function that returns the dyadic properties of two nodes.
    num_edges: int
        The total number of edges.
    num_edges_label: numpy.ndarray
        The number of edges per label.
    num_vertices: int
        The total number of nodes.
    num_labels: int
        The total number of labels.
    param: np.ndarray
        The parameter vector.
    per_label: bool
        Selects if the model has a parameter for each layer or just one.
    selfloops: bool
        Selects if self loops (connections from i to i) are allowed.

    Methods
    -------
    fit:
        Fit the parameters of the model with the given method.
    """

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, prop_out, prop_in):
        """Compute the probability of connection between node i and j.

        param is expected to be an array with num_labels elements. All
        properties must be a tuple (indices, values) from a sparse matrix.
        """
        # Check that parameter is not inf
        if isinf(d[0]):
            return 1.0, 0.0

        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
        x_lbl = prop_out[0]
        x_val = prop_out[1]
        y_lbl = prop_in[0]
        y_val = prop_in[1]
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = x_val[i] * y_val[j]
                    if isinf(tmp):
                        return 1.0, 0.0
                    else:
                        val += tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        val1 = d[0] * val
        if val1 == 0.0:
            return 0.0, val
        else:
            return -expm1(-val1), val * exp(-val1)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ijk(d, x_i, y_j):
        """Compute the probability of connection and the jacobian
        contribution of node i and j for layer k.
        """
        if (x_i == 0) or (y_j == 0):
            return 0.0, 0.0

        if d == 0:
            return 0.0, x_i * y_j

        tmp = x_i * y_j
        tmp1 = d * tmp
        if isinf(tmp1):
            return 1.0, 0.0
        else:
            return -expm1(-tmp1), tmp * exp(-tmp1)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(d, prop_out, prop_in, prop_dyad):
        """Compute the probability of connection between node i and j.

        param is expected to be an array with num_labels elements. All
        properties must be a tuple (indices, values) from a sparse matrix.
        """
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
        x_lbl = prop_out[0]
        x_val = prop_out[1]
        y_lbl = prop_in[0]
        y_val = prop_in[1]
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return 1.0
                    else:
                        val += tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        return -expm1(-val)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, prop_out, prop_in, prop_dyad):
        """Compute the log probability of connection between node i and j."""
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
        x_lbl = prop_out[0]
        x_val = prop_out[1]
        y_lbl = prop_in[0]
        y_val = prop_in[1]
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return 0.0
                    else:
                        val += tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        if val == 0.0:
            return -np.infty
        else:
            return log(-expm1(-val))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(d, prop_out, prop_in, prop_dyad):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
        x_lbl = prop_out[0]
        x_val = prop_out[1]
        y_lbl = prop_in[0]
        y_val = prop_in[1]
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return -np.infty
                    else:
                        val += tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        return -val

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ijk(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j on
        layer k.
        """
        if (x_i == 0) or (y_j == 0) or (d == 0):
            return 0.0

        tmp = d * x_i * y_j
        if isinf(tmp):
            return 1.0
        else:
            return -expm1(-tmp)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp_ijk(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j on
        layer k.
        """
        if (x_i == 0) or (y_j == 0) or (d == 0):
            return -np.infty

        tmp = d * x_i * y_j
        if isinf(tmp):
            return 0.0
        else:
            return log(-expm1(-tmp))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp_ijk(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j on
        layer k.
        """
        if (x_i == 0) or (y_j == 0) or (d == 0):
            return 0.0

        return -d * x_i * y_j
