from .ensembles import DiGraphEnsemble
import warnings
import numpy.random as rng
import scipy.sparse as sp
from . import graphs
import numpy as np
from numba import jit
from math import log
from math import log1p


class RandomDiGraph(DiGraphEnsemble):
    """An Erdős–Rényi random graph ensemble.

    Attributes
    ----------
    num_vertices: int
        The total number of vertices.
    num_edges: float
        The total number of edges.
    total_weight: float
        The sum of all edges weights.
    param: np.ndarray
        The parameters of the model. The first element is the probability of
        each link. The second element contains the parameter of the defining
        the probability distribution of weights.
    discrete_weights: boolean
        The flag determining if the distribution of weights is discrete or
        continuous.

    Methods
    -------
    fit:
        Fit the parameters of the model with the given method.
    """

    def __init__(self, *args, **kwargs):
        """Return a RandomGraph ensemble."""
        super().__init__(*args, **kwargs)

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.Graph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges()
                if g.weighted:
                    self.total_weight = g.total_weight()
            else:
                raise ValueError("First argument passed must be a Graph.")

            if len(args) > 1:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "param",
            "selfloops",
            "total_weight",
            "discrete_weights",
        ]
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError("Illegal argument passed: " + name)
            else:
                setattr(self, name, kwargs[name])

        # Check that all necessary attributes have been passed
        if not hasattr(self, "num_vertices"):
            raise ValueError("Number of vertices not set.")
        else:
            try:
                assert self.num_vertices / int(self.num_vertices) == 1
                self.num_vertices = int(self.num_vertices)
            except Exception:
                raise ValueError("Number of vertices must be an integer.")

            if self.num_vertices <= 0:
                raise ValueError("Number of vertices must be a positive number.")

        if not hasattr(self, "selfloops"):
            self.selfloops = False

        # Ensure that number of edges is a positive number
        if hasattr(self, "num_edges"):
            try:
                tmp = len(self.num_edges)
                if tmp == 1:
                    self.num_edges = self.num_edges[0]
                else:
                    raise ValueError("Number of edges must be a number.")
            except TypeError:
                pass

            try:
                self.num_edges = self.num_edges * 1.0
            except TypeError:
                raise ValueError("Number of edges must be a number.")

            if self.num_edges < 0:
                raise ValueError("Number of edges must be a positive number.")

        # Ensure that parameter is a single positive number
        if hasattr(self, "param"):
            if not isinstance(self.param, np.ndarray):
                self.param = np.array([self.param, 0])

            else:
                if not (len(self.param) == 2):
                    raise ValueError("The model requires two parameters.")

            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError("Parameters must be numeric.")

            if np.any(self.param < 0):
                raise ValueError("Parameters must be positive.")

        if not (hasattr(self, "num_edges") or hasattr(self, "param")):
            raise ValueError("Either num_edges or param must be set.")

        # Check if weight information is present
        if not hasattr(self, "discrete_weights") and hasattr(self, "total_weight"):
            self.discrete_weights = False

        # Ensure total weight is a number
        if hasattr(self, "total_weight"):
            try:
                tmp = len(self.total_weight)
                if tmp == 1:
                    self.total_weight = self.total_weight[0]
                else:
                    raise ValueError("Total weight must be a number.")
            except TypeError:
                pass

            try:
                self.total_weight = self.total_weight * 1.0
            except TypeError:
                raise ValueError("Total weight must be a number.")

            if self.total_weight < 0:
                raise ValueError("Total weight must be a positive number.")

    def fit(self):
        """Fit the parameter to the number of edges and total weight."""
        if self.selfloops:
            p = self.num_edges / self.num_vertices**2
        else:
            p = self.num_edges / (self.num_vertices * (self.num_vertices - 1))

        if hasattr(self, "total_weight"):
            if self.discrete_weights:
                q = 1 - self.num_edges / self.total_weight
            else:
                q = self.num_edges / self.total_weight
        else:
            q = 0

        self.param = np.array([p, q])

    def expected_num_edges(self, recompute=False):
        """Compute the expected number of edges (per label) given p."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if self.selfloops:
            self.exp_num_edges = self.param[0] * self.num_vertices**2
        else:
            self.exp_num_edges = (
                self.param[0] * self.num_vertices * (self.num_vertices - 1)
            )
        return self.exp_num_edges

    def expected_total_weight(self, recompute=False):
        """Compute the expected total weight (per label) given q."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if self.discrete_weights:
            self.exp_tot_weight = self.num_edges / (1 - self.q)
        else:
            self.exp_tot_weight = self.num_edges / self.q

        return self.exp_tot_weight

    def expected_degree(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        d = np.empty(self.num_vertices, dtype=np.float64)
        d[:] = (2 * self.param[0] - self.param[0] ** 2) * (self.num_vertices - 1)
        if self.selfloops:
            d[:] += self.param[0]

        return d

    def expected_out_degree(self, recompute=False):
        """Compute the expected out degree."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        d = np.empty(self.num_vertices, dtype=np.float64)
        if self.selfloops:
            d[:] = self.param[0] * self.num_vertices
        else:
            d[:] = self.param[0] * (self.num_vertices - 1)
        return d

    def expected_in_degree(self, recompute=False):
        """Compute the expected in degree."""
        return self.expected_out_degree(recompute=recompute)

    def expected_av_nn_property(
        self, prop, ndir="out", selfloops=False, deg_recompute=False
    ):
        """Computes the expected value of the nearest neighbour average of
        the property array. The array must have the first dimension
        corresponding to the vertex index.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        # Check first dimension of property array is correct
        if not prop.shape[0] == self.num_vertices:
            msg = (
                "Property array must have first dimension size be equal to"
                " the number of vertices."
            )
            raise ValueError(msg)

        # Set selfloops option
        tmp_self = self.selfloops
        if selfloops is None:
            selfloops = self.selfloops
        elif selfloops != self.selfloops:
            deg_recompute = True
            self.selfloops = selfloops

        # Compute correct expected degree
        if ndir == "out":
            deg = self.expected_out_degree(recompute=deg_recompute)
        elif ndir == "in":
            deg = self.expected_in_degree(recompute=deg_recompute)
        elif ndir == "out-in":
            deg = self.expected_degree(recompute=deg_recompute)
        else:
            raise ValueError("Neighbourhood direction not recognised.")

        # Compute av_nn_prop
        av_nn = np.empty(self.num_vertices, dtype=np.float64)
        if (ndir == "out") or (ndir == "in"):
            av_nn[:] = np.sum(self.param[0] * prop)
            if not self.selfloops:
                av_nn += -self.param[0] * prop
        elif ndir == "out-in":
            av_nn[:] = np.sum((2 * self.param[0] - self.param[0] ** 2) * prop)
            if not self.selfloops:
                av_nn += -(2 * self.param[0] - self.param[0] ** 2) * prop
        else:
            raise ValueError("Direction of neighbourhood not right.")

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        # Restore model self-loops properties if they have been modified
        if tmp_self != self.selfloops:
            self.selfloops = tmp_self

        return av_nn

    def log_likelihood(self, g, selfloops=None):
        """Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if selfloops is None:
            selfloops = self.selfloops

        if isinstance(g, graphs.Graph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(directed=True, weighted=False)
        elif sp.issparse(g):
            adj = g.asformat("csr")
        elif isinstance(g, np.ndarray):
            adj = sp.csr_array(g)
        else:
            raise ValueError("g input not a graph or adjacency matrix.")

        # Ensure dimensions are correct
        if adj.shape != (self.num_vertices, self.num_vertices):
            msg = (
                "Passed graph adjacency matrix does not have the correct "
                "shape: {0} instead of {1}".format(
                    adj.shape, (self.num_vertices, self.num_vertices)
                )
            )
            raise ValueError(msg)

        # Compute log likelihood of graph
        if (self.param[0] == 0) and (adj.nnz > 0):
            return -np.infty
        if (self.param[0] == 1) and (adj.nnz != 0):
            return -np.infty

        like = adj.nnz * log(self.param[0])
        if selfloops:
            like += (self.num_vertices**2 - adj.nnz) * log1p(-self.param[0])
        else:
            like += (self.num_vertices * (self.num_vertices - 1) - adj.nnz) * log1p(
                -self.param[0]
            )
            # Ensure that the matrix has no elements on the diagonal
            if adj.diagonal().sum() > 0:
                return -np.infty

        return like

    def sample(
        self,
        ref_g=None,
        weights=None,
        out_strength=None,
        in_strength=None,
        selfloops=None,
    ):
        """Return a Graph sampled from the ensemble.

        If a reference graph is passed (ref_g) then the properties of the graph
        will be copied to the new samples.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before sampling.")

        if selfloops is None:
            selfloops = self.selfloops

        # Generate uninitialised graph object
        g = graphs.DiGraph.__new__(graphs.DiGraph)

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = g.get_num_bytes(g.num_vertices)
        g.id_dtype = np.dtype("u" + str(num_bytes))

        # Check if reference graph is available
        if ref_g is not None:
            if hasattr(ref_g, "num_groups"):
                g.num_groups = ref_g.num_groups
                g.group_dict = ref_g.group_dict
                g.group_dtype = ref_g.group_dtype
                g.groups = ref_g.groups

            g.id_dict = ref_g.id_dict
        else:
            g.id_dict = {}
            for i in range(g.num_vertices):
                g.id_dict[i] = i

        # Sample edges
        if weights is None:
            rows, cols = self._binary_sample(
                self.param, self.num_vertices, self.selfloops
            )
            vals = np.ones(len(rows), dtype=bool)
        elif weights == "random":
            if self.discrete_weights:
                rows, cols, vals = self._discrete_weighted_sample(
                    self.param, self.num_vertices, self.selfloops
                )
            else:
                rows, cols, vals = self._weighted_sample(
                    self.param, self.num_vertices, self.selfloops
                )
        elif weights == "cremb":
            if out_strength is None:
                out_strength = self.prop_out
            if in_strength is None:
                in_strength = self.prop_in
            rows, cols, vals = self._cremb_sample(
                self.param, self.num_vertices, out_strength, in_strength, self.selfloops
            )
        else:
            raise ValueError("Weights method not recognised or implemented.")

        # Convert to adjacency matrix
        g.adj = sp.csr_array(
            (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
        )

        return g

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _binary_sample(param, num_v, selfloops):
        """Sample from the ensemble."""
        rows = []
        cols = []
        p = param[0]
        for i in range(num_v):
            for j in range(num_v):
                if (i != j) | selfloops:
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _weighted_sample(param, num_v, selfloops):
        """Sample from the ensemble."""
        rows = []
        cols = []
        vals = []
        p = param[0]
        for i in range(num_v):
            for j in range(num_v):
                if (i != j) | selfloops:
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        vals.append(rng.exponential(1 / param[1]))

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _discrete_weighted_sample(param, num_v, selfloops):
        """Sample from the ensemble."""
        rows = []
        cols = []
        vals = []
        p = param[0]
        for i in range(num_v):
            for j in range(num_v):
                if (i != j) | selfloops:
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        vals.append(rng.geometric(1 - param[1]))

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample(param, num_v, s_out, s_in, selfloops):
        """Sample from the ensemble with weights from the CremB model."""
        s_tot = np.sum(s_out)
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in) / s_tot) < 1e-6, msg
        rows = []
        cols = []
        vals = []
        rows = []
        cols = []
        vals = []
        p = param[0]
        for i in range(num_v):
            for j in range(num_v):
                if (i != j) | selfloops:
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        vals.append(rng.exponential(s_out[i] * s_in[j] / (s_tot * p)))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(param, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j."""
        return param[0]

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(param, x_i, y_j, z_ij):
        """Compute the log probability of connection between node i and j."""
        return log(param[0])

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(param, x_i, y_j, z_ij):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        return log1p(-param[0])
