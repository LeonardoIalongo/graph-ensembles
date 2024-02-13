from .invariant import ScaleInvariantModel
from .. import graphs
from .ensembles import empty_index
from ...solver import monotonic_newton_solver
import numpy as np
import scipy.sparse as sp
import numpy.random as rng
from numba import jit
from math import isinf
from math import expm1
from math import exp
import warnings


class ConditionalInvariantModel(ScaleInvariantModel):
    """The Conditional Invariant model is a ScaleInvariantModel that is
    conditioned on the observation of a coarse grained graph. This changes the
    probability distribution and the computation of certain properties.


    Attributes
    ----------
    prop_out: np.ndarray
        The out fitness sequence.
    prop_in: np.ndarray
        the in fitness sequence.
    prop_dyad: function
        A function that returns the dyadic properties of two nodes.
    groups: np.ndarray
        An array that returns the macro-node to which each node belongs to.
    adj:
        The coarse-grained adjacency matrix.
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
        """Return a ConditionalScaleInvariantModel for the given graph data.

        The model accepts as arguments either: two DiGraphs, in which case the
        strengths are used as fitnesses, or directly the fitness sequences (in
        and out). The model accepts the fitness sequences as numpy arrays. The
        two DiGraphs are assumed to be respectively the micro and observed macro.

        In order to do the conditioning the model requires the groups array,
        which should be a numpy array with the group identifier, and the
        adjacency matrix of the observed coarse-grained graph. This last matrix
        can be given as a dense or sparse matrix.


        """
        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.DiGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges()
                self.prop_out = g.out_strength()
                self.prop_in = g.in_strength()
                if hasattr(g, "groups"):
                    self.groups = g.groups
            else:
                raise ValueError("First argument passed must be a " "DiGraph.")

            if len(args) > 1:
                if isinstance(args[1], graphs.DiGraph):
                    g = args[1]
                    self.adj = g.adjacency_matrix(directed=True, weighted=True)
                elif isinstance(args[1], np.ndarray):
                    self.adj = args[1]
                elif sp.issparse(args[1]):
                    self.adj = args[1]
                else:
                    raise ValueError(
                        "Second argument passed must be a DiGraph or an adjacency matrix."
                    )

            if len(args) > 2:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "prop_out",
            "prop_in",
            "groups",
            "adj",
            "param",
            "selfloops",
        ]
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError("Illegal argument passed: " + name)
            else:
                setattr(self, name, kwargs[name])

        # Ensure that all necessary fields have been set
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

        if not hasattr(self, "prop_out"):
            raise ValueError("prop_out not set.")
        elif isinstance(self.prop_out, empty_index):
            raise ValueError("prop_out not set.")

        if not hasattr(self, "prop_in"):
            raise ValueError("prop_in not set.")
        elif isinstance(self.prop_in, empty_index):
            raise ValueError("prop_in not set.")

        if not hasattr(self, "selfloops"):
            self.selfloops = False

        # Ensure that fitnesses passed adhere to format (ndarray)
        msg = "Node out properties must be a numpy array of length " + str(
            self.num_vertices
        )
        assert isinstance(self.prop_out, np.ndarray), msg
        assert self.prop_out.shape == (self.num_vertices,), msg

        msg = "Node in properties must be a numpy array of length " + str(
            self.num_vertices
        )
        assert isinstance(self.prop_in, np.ndarray), msg
        assert self.prop_in.shape == (self.num_vertices,), msg

        # Ensure that fitnesses have positive values only
        msg = "Node out properties must contain positive values only."
        assert np.all(self.prop_out >= 0), msg

        msg = "Node in properties must contain positive values only."
        assert np.all(self.prop_in >= 0), msg

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
                self.param = np.array([self.param])

            else:
                if not (len(self.param) == 1):
                    raise ValueError("The model requires one parameter.")

            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError("Parameters must be numeric.")

            if np.any(self.param < 0):
                raise ValueError("Parameters must be positive.")

        if not (hasattr(self, "num_edges") or hasattr(self, "param")):
            raise ValueError("Either num_edges or param must be set.")

        # Ensure that the number of groups and adj have been set
        if not hasattr(self, "adj") or not hasattr(self, "groups"):
            raise ValueError(
                "Both the adjacency matrix of the observed graph "
                " and the partitioning groups of the graph must be set."
            )

        # If adj is sparse set to dense
        if not sp.issparse(self.adj):
            self.adj = sp.csr_array(self.adj)

        # Check that the two are consistent
        self.num_groups = self.adj.shape[0]
        assert self.adj.ndim == 2, "Adjacency matrix must be 2 dimensional."
        assert self.adj.shape[1] == self.num_groups, "Adjacency matrix must be square."
        assert isinstance(
            self.groups[0], (int, np.integer)
        ), "groups array must be of integers."
        assert (
            np.max(self.groups) + 1 == self.num_groups
        ), "Number of groups in adj and groups array does not match."

    def expected_num_edges(self, recompute=False):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges") or recompute:
            self._exp_num_edges = self.exp_edges(
                self.cond_p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.groups,
                self.adj.indptr,
                self.adj.indices,
                self.selfloops,
            )

        return self._exp_num_edges

    def expected_degree(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_degree") or recompute:
            res = self.exp_degrees(
                self.cond_p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.groups,
                self.adj.indptr,
                self.adj.indices,
                self.selfloops,
            )
            self._exp_degree = res[0]
            self._exp_out_degree = res[1]
            self._exp_in_degree = res[2]

        return self._exp_degree

    def expected_out_degree(self, recompute=False):
        """Compute the expected out degree."""
        if not hasattr(self, "_exp_out_degree") or recompute:
            _ = self.expected_degree(recompute=recompute)

        return self._exp_out_degree

    def expected_in_degree(self, recompute=False):
        """Compute the expected in degree."""
        if not hasattr(self, "_exp_in_degree") or recompute:
            _ = self.expected_degree(recompute=recompute)

        return self._exp_in_degree

    def log_likelihood(self, g):
        """Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

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

        # Check if adjacency matrix compatible with aggregate conditional
        M = self.groups.max() + 1
        N = self.num_vertices
        Gmat = sp.csr_array((np.ones(N), (self.groups, np.arange(N))), shape=(M, N))
        agg_adj = Gmat.dot(adj).dot(Gmat.T) > 0
        if (agg_adj != (self.adj > 0)).nnz != 0:
            return -np.infty

        # Compute log likelihood of graph
        like = self._likelihood(
            self.logp,
            self.log1mp,
            self.param,
            self.prop_out,
            self.prop_in,
            self.prop_dyad,
            adj.indptr,
            adj.indices,
            self.selfloops,
        )

        # Compute log likelihood of agg graph
        agg_like = self._agg_likelihood(
            self.logp,
            self.log1mp,
            self.param,
            self.prop_out,
            self.prop_in,
            self.groups,
            agg_adj.indptr,
            agg_adj.indices,
            self.selfloops,
        )

        return like - agg_like

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
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.groups,
                self.adj.indptr,
                self.adj.indices,
                self.selfloops,
            )
            vals = np.ones(len(rows), dtype=bool)
        elif weights == "cremb":
            if out_strength is None:
                out_strength = self.prop_out
            if in_strength is None:
                in_strength = self.prop_in
            rows, cols, vals = self._cremb_sample(
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.groups,
                self.adj.indptr,
                self.adj.indices,
                out_strength,
                in_strength,
                self.selfloops,
            )
        elif weights == "wagg":
            if out_strength is None:
                out_strength = self.prop_out
            if in_strength is None:
                in_strength = self.prop_in
            rows, cols, vals = self._wagg_sample(
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.groups,
                self.adj.indptr,
                self.adj.indices,
                self.adj.data,
                out_strength,
                in_strength,
                self.selfloops,
            )
        else:
            raise ValueError("Weights method not recognised or implemented.")

        # Convert to adjacency matrix
        g.adj = sp.csr_array(
            (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
        )

        return g

    def fit(
        self,
        x0=None,
        method="density",
        atol=1e-24,
        rtol=1e-9,
        maxiter=100,
        verbose=False,
    ):
        """Fit the parameter either to match the given number of edges or
            using maximum likelihood estimation.

        Parameters
        ----------
        x0: float
            Optional initial conditions for parameters.
        method: 'density' or 'mle'
            Selects whether to fit param using maximum likelihood estimation
            or by ensuring that the expected density matches the given one.
        atol : float
            Absolute tolerance for the exit condition.
        rtol : float
            Relative tolerance for the exit condition.
        max_iter : int or float
            Maximum number of iteration.
        verbose: boolean
            If true print debug info while iterating.
        """
        if x0 is None:
            x0 = np.array([0], dtype=np.float64)

        if not isinstance(x0, np.ndarray):
            x0 = np.array([x0])

        if not (len(x0) == 1):
            raise ValueError("The model requires one parameter.")

        if not np.issubdtype(x0.dtype, np.number):
            raise ValueError("x0 must be numeric.")

        if np.any(x0 < 0):
            raise ValueError("x0 must be positive.")

        if method == "density":
            # Ensure that num_edges is set
            if not hasattr(self, "num_edges"):
                raise ValueError("Number of edges must be set for density solver.")
            sol = monotonic_newton_solver(
                x0,
                self.density_fit_fun,
                self.num_edges,
                atol=atol,
                rtol=rtol,
                x_l=0.0,
                x_u=np.infty,
                max_iter=maxiter,
                full_return=True,
                verbose=verbose,
            )

        elif method == "mle":
            raise ValueError("Method not implemented.")

        else:
            raise ValueError("The selected method is not valid.")

        # Update results and check convergence
        self.param = sol.x
        self.solver_output = sol

        if not self.solver_output.converged:
            warnings.warn("Fit did not converge", UserWarning)

    def density_fit_fun(self, delta):
        """Return the objective function value and the Jacobian
        for a given value of delta.
        """
        f, jac = self.exp_edges_f_jac(
            self.p_jac_ij,
            delta,
            self.prop_out,
            self.prop_in,
            self.groups,
            self.adj.indptr,
            self.adj.indices,
            self.selfloops,
        )

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges(p_ij, param, prop_out, prop_in, groups, adj_i, adj_j, selfloops):
        """Compute the expected number of edges."""
        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        # Compute expected edges
        exp_e = 0.0
        for i, x_i in enumerate(prop_out):
            a = groups[i]
            n = adj_i[a]
            m = adj_i[a + 1]
            gr_list = adj_j[n:m]
            for j, y_j in enumerate(prop_in):
                b = groups[j]
                if (b in gr_list) and ((i != j) or selfloops):
                    if not selfloops and (a == b):
                        pij = p_ij(
                            param,
                            x_i,
                            y_j,
                            1.0,
                            agg_p_out[a],
                            agg_p_in[b],
                            1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                        )
                    else:
                        pij = p_ij(
                            param,
                            x_i,
                            y_j,
                            1.0,
                            agg_p_out[a],
                            agg_p_in[b],
                            1.0,
                        )
                    exp_e += pij

        return exp_e

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_degrees(p_ij, param, prop_out, prop_in, groups, adj_i, adj_j, selfloops):
        """Compute the expected undirected, in and out degree sequences."""
        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        # Compute expected degrees
        num_v = len(groups)
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)

        for i, p_out_i in enumerate(prop_out):
            p_in_i = prop_in[i]
            a = groups[i]
            n = adj_i[a]
            m = adj_i[a + 1]
            gr_list = adj_j[n:m]
            for j in range(i + 1):
                b = groups[j]
                if b in gr_list:
                    p_out_j = prop_out[j]
                    p_in_j = prop_in[j]
                    if i != j:
                        if not selfloops and (a == b):
                            pij = p_ij(
                                param,
                                p_out_i,
                                p_in_j,
                                1.0,
                                agg_p_out[a],
                                agg_p_in[b],
                                1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                            )
                            pji = p_ij(
                                param,
                                p_out_j,
                                p_in_i,
                                1.0,
                                agg_p_out[b],
                                agg_p_in[a],
                                1 - agg_diag[a] / (agg_p_out[b] * agg_p_in[a]),
                            )
                        else:
                            pij = p_ij(
                                param,
                                p_out_i,
                                p_in_j,
                                1.0,
                                agg_p_out[a],
                                agg_p_in[b],
                                1.0,
                            )
                            pji = p_ij(
                                param,
                                p_out_j,
                                p_in_i,
                                1.0,
                                agg_p_out[b],
                                agg_p_in[a],
                                1.0,
                            )
                        p = pij + pji - pij * pji
                        exp_d[i] += p
                        exp_d[j] += p
                        exp_d_out[i] += pij
                        exp_d_out[j] += pji
                        exp_d_in[j] += pij
                        exp_d_in[i] += pji
                    elif selfloops:
                        pii = p_ij(
                            param,
                            p_out_i,
                            p_in_j,
                            1.0,
                            agg_p_out[groups[i]],
                            agg_p_in[groups[j]],
                            1.0,
                        )
                        exp_d[i] += pii
                        exp_d_out[i] += pii
                        exp_d_in[j] += pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _binary_sample(p_ij, param, prop_out, prop_in, groups, adj_i, adj_j, selfloops):
        """Sample from the ensemble."""
        rows = []
        cols = []

        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        # Iterate over aggregated adj
        for a in range(M):
            # Get all node indices in a
            i_in_a = np.where(groups == a)[0]

            # For each group find connected groups
            n = adj_i[a]
            m = adj_i[a + 1]
            b_list = adj_j[n:m]

            for b in b_list:
                # Get all node indices in b
                j_in_b = np.where(groups == b)[0]

                # Iterate in order over pij that compose A_ab
                atleastone = False
                pnorm = p_ij(param, agg_p_out[a], agg_p_in[b], 1)
                if not selfloops and (a == b):
                    pnorm = p_ij(
                        param,
                        agg_p_out[a],
                        agg_p_in[b],
                        1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                    )

                for i in i_in_a:
                    p_out_i = prop_out[i]
                    for j in j_in_b:
                        if (i != j) | selfloops:
                            p = p_ij(param, p_out_i, prop_in[j], 1)

                            if atleastone:
                                p_sample = p
                            else:
                                p_sample = p / pnorm

                            if rng.random() < p_sample:
                                rows.append(i)
                                cols.append(j)
                                atleastone = True
                            else:
                                pnorm = 1 - ((1 - pnorm) / (1 - p))

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample(
        p_ij,
        param,
        prop_out,
        prop_in,
        groups,
        adj_i,
        adj_j,
        s_out,
        s_in,
        selfloops,
    ):
        """Sample from the ensemble."""
        rows = []
        cols = []
        vals = []

        # Compute total strength
        s_tot = np.sum(s_out)
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in) / s_tot) < 1e-6, msg

        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        # Iterate over aggregated adj
        for a in range(M):
            # Get all node indices in a
            i_in_a = np.where(groups == a)[0]

            # For each group find connected groups
            n = adj_i[a]
            m = adj_i[a + 1]
            b_list = adj_j[n:m]

            for b in b_list:
                # Get all node indices in b
                j_in_b = np.where(groups == b)[0]

                # Iterate in order over pij that compose A_ab
                atleastone = False
                p_ab = p_ij(param, agg_p_out[a], agg_p_in[b], 1)
                if not selfloops and (a == b):
                    p_ab = p_ij(
                        param,
                        agg_p_out[a],
                        agg_p_in[b],
                        1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                    )
                pnorm = p_ab

                for i in i_in_a:
                    p_out_i = prop_out[i]
                    for j in j_in_b:
                        if (i != j) | selfloops:
                            p = p_ij(param, p_out_i, prop_in[j], 1)

                            if atleastone:
                                p_sample = p
                            else:
                                p_sample = p / pnorm

                            if rng.random() < p_sample:
                                rows.append(i)
                                cols.append(j)
                                vals.append(
                                    rng.exponential(
                                        s_out[i] * s_in[j] / (s_tot * (p / p_ab))
                                    )
                                )

                                atleastone = True
                            else:
                                pnorm = 1 - ((1 - pnorm) / (1 - p))

        return rows, cols, vals

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _wagg_sample(
        p_ij,
        param,
        prop_out,
        prop_in,
        groups,
        adj_i,
        adj_j,
        adj_v,
        s_out,
        s_in,
        selfloops,
    ):
        """Sample from the ensemble."""
        rows = []
        cols = []
        vals = []

        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        # Iterate over aggregated adj
        for a in range(M):
            # Get all node indices in a
            i_in_a = np.where(groups == a)[0]

            # For each group find connected groups
            n = adj_i[a]
            m = adj_i[a + 1]
            b_list = adj_j[n:m]

            # Get w_ab total flows
            w_list = adj_v[n:m]

            for b, w_ab in zip(b_list, w_list):
                # Get all node indices in b
                j_in_b = np.where(groups == b)[0]

                # Iterate in order over pij that compose A_ab
                atleastone = False
                pnorm = p_ij(param, agg_p_out[a], agg_p_in[b], 1)
                if not selfloops and (a == b):
                    pnorm = p_ij(
                        param,
                        agg_p_out[a],
                        agg_p_in[b],
                        1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                    )

                # Keep track of new samples to sample the weights
                s_start = len(rows)
                snorm = 0
                for i in i_in_a:
                    p_out_i = prop_out[i]
                    for j in j_in_b:
                        if (i != j) | selfloops:
                            p = p_ij(param, p_out_i, prop_in[j], 1)

                            if atleastone:
                                p_sample = p
                            else:
                                p_sample = p / pnorm

                            if rng.random() < p_sample:
                                rows.append(i)
                                cols.append(j)
                                snorm += s_out[i] * s_in[j]
                                vals.append(s_out[i] * s_in[j])
                                atleastone = True
                            else:
                                pnorm = 1 - ((1 - pnorm) / (1 - p))

                # Iterate over the samples of macro link ab
                for n in range(s_start, len(rows)):
                    frac = vals[n] / snorm
                    vals[n] = frac * w_ab

        return rows, cols, vals

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac(
        p_jac_ij, param, prop_out, prop_in, groups, adj_i, adj_j, selfloops
    ):
        """Compute the objective function of the density solver and its
        derivative.
        """
        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        f = 0.0
        jac = 0.0
        for i, p_out_i in enumerate(prop_out):
            a = groups[i]
            n = adj_i[a]
            m = adj_i[a + 1]
            gr_list = adj_j[n:m]
            for j, p_in_j in enumerate(prop_in):
                b = groups[j]
                if (b in gr_list) and ((i != j) | selfloops):
                    if not selfloops and (a == b):
                        p_tmp, jac_tmp = p_jac_ij(
                            param,
                            p_out_i,
                            p_in_j,
                            1.0,
                            agg_p_out[a],
                            agg_p_in[b],
                            1 - agg_diag[a] / (agg_p_out[a] * agg_p_in[b]),
                        )
                    else:
                        p_tmp, jac_tmp = p_jac_ij(
                            param, p_out_i, p_in_j, 1.0, agg_p_out[a], agg_p_in[b], 1.0
                        )

                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, x_i, y_j, z_ij, x_I, y_J, z_IJ):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        if (
            (x_I == 0)
            or (y_J == 0)
            or (z_IJ == 0)
            or (x_i == 0)
            or (y_j == 0)
            or (z_ij == 0)
        ):
            return 0.0, 0.0

        a = x_i * y_j * z_ij
        b = x_I * y_J * z_IJ
        da = d[0] * a
        db = d[0] * b
        expdbm1 = -expm1(-db)

        if expdbm1 < 1e-12:
            return (
                (x_i / x_I) * (y_j / y_J) * (z_ij / z_IJ),
                (1 / 2 - (a / (2 * b))) * a,
            )

        if isinf(da):
            return 1.0, 0.0
        elif isinf(db):
            return -expm1(-da), a * exp(-da)
        else:
            return (
                -expm1(-da) / expdbm1,
                a * exp(-da) / expdbm1 - b * exp(-db) * (-expm1(-da)) / (expdbm1**2),
            )

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def cond_p_ij(d, x_i, y_j, z_ij, x_I, y_J, z_IJ):
        """Compute the probability of connection between node i and j."""
        if (
            (x_I == 0)
            or (y_J == 0)
            or (z_IJ == 0)
            or (x_i == 0)
            or (y_j == 0)
            or (z_ij == 0)
        ):
            return 0.0

        if d[0] == 0:
            return (x_i / x_I) * (y_j / y_J) * (z_ij / z_IJ)

        tmp = d[0] * x_i * y_j * z_ij
        tmp1 = d[0] * x_I * y_J * z_IJ
        if isinf(tmp):
            return 1.0
        elif isinf(tmp1):
            return -expm1(-tmp)
        else:
            return (-expm1(-tmp)) / (-expm1(-tmp1))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def prop_dyad(i, j):
        """Define empy dyadic property as it is not always defined."""
        return 1.0

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _agg_likelihood(
        logp,
        log1mp,
        param,
        prop_out,
        prop_in,
        groups,
        adj_i,
        adj_j,
        selfloops,
    ):
        # Compute aggregate properties
        M = np.max(groups) + 1
        agg_p_out = np.zeros(M, np.float64)
        agg_p_in = np.zeros(M, np.float64)
        if not selfloops:
            agg_diag = np.zeros(M, np.float64)
        for i, gr in enumerate(groups):
            agg_p_out[gr] += prop_out[i]
            agg_p_in[gr] += prop_in[i]
            if not selfloops:
                agg_diag[gr] += prop_out[i] * prop_in[i]

        like = 0
        for i, p_out_i in enumerate(agg_p_out):
            n = adj_i[i]
            m = adj_i[i + 1]
            j_list = adj_j[n:m]
            for j, p_in_j in enumerate(agg_p_in):
                if not selfloops and (i == j):
                    # Check if link exists
                    if j in j_list:
                        tmp = logp(
                            param,
                            p_out_i,
                            p_in_j,
                            1 - agg_diag[i] / (agg_p_out[i] * agg_p_in[j]),
                        )
                    else:
                        tmp = log1mp(
                            param,
                            p_out_i,
                            p_in_j,
                            1 - agg_diag[i] / (agg_p_out[i] * agg_p_in[j]),
                        )
                else:
                    # Check if link exists
                    if j in j_list:
                        tmp = logp(param, p_out_i, p_in_j, 1.0)
                    else:
                        tmp = log1mp(param, p_out_i, p_in_j, 1.0)

                if isinf(tmp):
                    return tmp
                like += tmp

        return like
