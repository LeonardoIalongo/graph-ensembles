""" This module defines the classes that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """
from ..sparse import graphs
from ..solver import monotonic_newton_solver
import numpy as np
import numpy.random as rng
import scipy.sparse as sp
import warnings
from numba import jit
from math import floor
from math import exp
from math import expm1
from math import log
from math import log1p
from math import isinf
from pyspark import SparkContext
from numba import float64
from numba.experimental import jitclass
from numba.typed import List


# Define a special empty iterator class. This is necessary if the model does
# not have node properties.
spec = [
    ("array", float64[:]),
]


@jitclass(spec)  # pragma: no cover
class empty_index:
    def __init__(self):
        pass

    def __getitem__(self, index):
        return 1.0

    def max(self):
        return 1.0

    def min(self):
        return 1.0


# Global function definitions
@jit(nopython=True)  # pragma: no cover
def in_range(i, ind, fold):
    if fold:
        return range(i + 1)
    else:
        return range(ind[1] - ind[0])


class GraphEnsemble:
    """General class for Graph ensembles.

    All ensembles can be defined in three ways:

    1) From a suitable Graph object: we can think this as a randomization of
    the observed graph. The conserved quantities and relevant vertex
    attributes are computed on the original graph to initialise the ensemble.
    It is then possible to fit the model parameters in order to get a
    probability distribution over all graphs from which to sample.

    2) From conserved quantities and relevant vertex attributes directly: in
    the case we do not have a reference graph but we do know what properties
    we want the ensemble to hold, we can directly use those properties to
    initialise the model. Once this step is completed we can similarly fit the
    parameters and sample from the ensemble.

    3) Fully specifying all model parameters: a final possibility is to
    initialise the model by giving it the list of parameters it needs in order
    to define the probability distribution over graphs. In this case we do not
    need to fit the model and the value of the conserved quantities over the
    ensemble will depend on the parameters passed to the model rather than
    vice versa.

    What these three possibilities entail will depend on the specifics of the
    model.

    Note that if keyword arguments are passed together with a Graph, then the
    arguments overwrite the graph property. This allows for easier definition
    of the ensemble for example when we want to modify one aspect of the
    reference graph but not all (e.g. only the density, but keeping strengths
    the same).

    """

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def prop_dyad(i, j):
        """Define empy dyadic property as it is not always defined."""
        return None


class DiGraphEnsemble(GraphEnsemble):
    """General class for DiGraph ensembles.

    All ensembles are assumed to have independent edges whose probabilities
    depend only on a set of parameters (param), a set of node specific out and
    in properties (prop_out and prop_in), and a set of dyadic properties
    (prop_dyad). The ensemble is defined by the probability function
    pij(param, prop_out, prop_in, prop_dyad).

    Methods
    -------
    expected_num_edges:
        Compute the expected number of edges in the ensemble.
    expected_degree:
        Compute the expected undirected degree of each node.
    expected_out_degree:
        Compute the expected out degree of each node.
    expected_in_degree:
        Compute the expected in degree of each node.
    expected_av_nn_property:
        Compute the expected average of the given property of the nearest
        neighbours of each node.
    expected_av_nn_degree:
        Compute the expected average of the degree of the nearest
        neighbours of each node.
    log_likelihood:
        Compute the likelihood of the given graph.
    sample:
        Return a sample from the ensemble.

    """

    def __init__(self, *args, **kwargs):
        self.prop_out = empty_index()
        self.prop_in = empty_index()

    def expected_num_edges(self, recompute=False):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges") or recompute:
            # It is necessary to select the elements or pickling will fail
            e_fun = self.exp_edges
            p_ij = self.p_ij
            delta = self.param
            pdyad = self.prop_dyad
            slflp = self.selfloops
            tmp = self.p_iter_rdd.map(
                lambda x: e_fun(p_ij, delta, x[0][0], x[0][1], x[1], x[2], pdyad, slflp)
            )
            self._exp_num_edges = tmp.fold(0, lambda x, y: x + y)

        return self._exp_num_edges

    def expected_degree(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if not hasattr(self, "_exp_degree") or recompute:
            # It is necessary to select the elements or pickling will fail
            e_fun = self.exp_degrees
            p_ij = self.p_ij
            delta = self.param
            pdyad = self.prop_dyad
            slflp = self.selfloops
            num_v = self.num_vertices
            tmp = self.p_sym_rdd.map(
                lambda x: e_fun(
                    p_ij, delta, x[0][0], x[0][1], x[1], x[2], pdyad, num_v, slflp
                )
            )
            exp_d = np.zeros(num_v, dtype=np.float64)
            exp_d_out = np.zeros(num_v, dtype=np.float64)
            exp_d_in = np.zeros(num_v, dtype=np.float64)
            res = tmp.fold(
                (exp_d, exp_d_out, exp_d_in),
                lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
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

        # It is necessary to select the elements or pickling will fail
        e_fun = self.exp_av_nn_prop
        p_ij = self.p_ij
        delta = self.param
        pdyad = self.prop_dyad
        tmp = self.p_sym_rdd.map(
            lambda x: e_fun(
                p_ij, delta, x[0][0], x[0][1], x[1], x[2], pdyad, prop, ndir, selfloops
            )
        )
        av_nn = tmp.fold(np.zeros(prop.shape, dtype=np.float64), lambda x, y: x + y)

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        # Restore model self-loops properties if they have been modified
        if tmp_self != self.selfloops:
            self.selfloops = tmp_self
            if hasattr(self, "_exp_out_degree"):
                del self._exp_out_degree
            if hasattr(self, "_exp_in_degree"):
                del self._exp_in_degree
            if hasattr(self, "_exp_degree"):
                del self._exp_degree

        return av_nn

    def expected_av_nn_degree(
        self,
        ddir="out",
        ndir="out",
        selfloops=False,
        deg_recompute=False,
        recompute=None,
    ):
        """Computes the expected value of the nearest neighbour average of
        the degree.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        # Compute property name
        name = "exp_av_" + ndir.replace("-", "_") + "_nn_d_" + ddir.replace("-", "_")

        if not hasattr(self, name) or recompute or deg_recompute:
            # Set selfloops option
            tmp_self = self.selfloops
            if selfloops is None:
                selfloops = self.selfloops
            elif selfloops != self.selfloops:
                deg_recompute = True
                self.selfloops = selfloops

            # Compute correct expected degree
            if ddir == "out":
                deg = self.expected_out_degree(recompute=deg_recompute)
            elif ddir == "in":
                deg = self.expected_in_degree(recompute=deg_recompute)
            elif ddir == "out-in":
                deg = self.expected_degree(recompute=deg_recompute)
            else:
                raise ValueError("Degree type not recognised.")

            # Compute property and set attribute
            res = self.expected_av_nn_property(
                deg, ndir=ndir, selfloops=selfloops, deg_recompute=False
            )
            setattr(self, name, res)

            # Restore model self-loops properties if they have been modified
            if tmp_self != self.selfloops:
                self.selfloops = tmp_self
                if hasattr(self, "_exp_out_degree"):
                    del self._exp_out_degree
                if hasattr(self, "_exp_in_degree"):
                    del self._exp_in_degree
                if hasattr(self, "_exp_degree"):
                    del self._exp_degree

        return getattr(self, name)

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
            adj = sp.csr_matrix(g)
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
        e_fun = self._likelihood
        logp = self.logp
        log1mp = self.log1mp
        delta = self.param
        pdyad = self.prop_dyad
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(
                logp,
                log1mp,
                delta,
                x[0][0],
                x[0][1],
                x[1],
                x[2],
                pdyad,
                adj.indptr,
                adj.indices,
                selfloops,
            )
        )
        like = tmp.fold(0, lambda x, y: x + y)

        return like

    def confusion_matrix(self, g, thresholds=None, selfloops=None):
        """Compute the true/false positive/negative rates for the model
        at the given probability thresholds levels.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if selfloops is None:
            selfloops = self.selfloops

        # If thresholds are not given or is an int compute automatically
        if thresholds is None:
            p_max = self.p_ij(self.param, self.prop_out.max(), self.prop_in.max(), None)
            p_min = self.p_ij(self.param, self.prop_out.min(), self.prop_in.min(), None)
            thresholds = np.logspace(log(p_min), log(p_max), 20)

        elif isinstance(thresholds, int):
            p_max = self.p_ij(self.param, self.prop_out.max(), self.prop_in.max(), None)
            p_min = self.p_ij(self.param, self.prop_out.min(), self.prop_in.min(), None)
            thresholds = np.logspace(log(p_min), log(p_max), thresholds)

        elif not isinstance(thresholds, np.ndarray):
            raise ValueError("Thresholds must be an array or an integer.")

        if isinstance(g, graphs.Graph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(directed=True, weighted=False)
        elif sp.issparse(g):
            adj = g.asformat("csr")
        elif isinstance(g, np.ndarray):
            adj = sp.csr_matrix(g)
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
        e_fun = self._confusion_matrix
        p = self.p_ij
        delta = self.param
        pdyad = self.prop_dyad
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(
                thresholds,
                p,
                delta,
                x[0][0],
                x[0][1],
                x[1],
                x[2],
                pdyad,
                adj.indptr,
                adj.indices,
                selfloops,
            )
        )
        tp = np.zeros(thresholds.shape, dtype=np.int64)
        fp = np.zeros(thresholds.shape, dtype=np.int64)
        tn = np.zeros(thresholds.shape, dtype=np.int64)
        fn = np.zeros(thresholds.shape, dtype=np.int64)
        res = tmp.fold(
            (tp, fp, tn, fn),
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]),
        )

        return res[0], res[1], res[2], res[3]

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
        p_ij = self.p_ij
        delta = self.param
        pdyad = self.prop_dyad
        app_fun = self.tuple_of_lists_append
        if weights is None:
            e_fun = self._binary_sample
            tmp = self.p_iter_rdd.map(
                lambda x: e_fun(
                    p_ij, delta, x[0][0], x[0][1], x[1], x[2], pdyad, selfloops
                )
            )
            rows, cols = tmp.fold(([], []), lambda x, y: app_fun(x, y))
            vals = np.ones(len(rows), dtype=bool)

        elif weights == "cremb":
            if out_strength is None:
                out_strength = self.prop_out
            if in_strength is None:
                in_strength = self.prop_in
            e_fun = self._cremb_sample
            tmp = self.p_iter_rdd.map(
                lambda x: e_fun(
                    p_ij,
                    delta,
                    x[0][0],
                    x[0][1],
                    x[1],
                    x[2],
                    pdyad,
                    out_strength,
                    in_strength,
                    selfloops,
                )
            )
            rows, cols, vals = tmp.fold(([], [], []), lambda x, y: app_fun(x, y))

        else:
            raise ValueError("Weights method not recognised or implemented.")

        # Convert to adjacency matrix
        g.adj = sp.csr_matrix(
            (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
        )

        return g

    @staticmethod
    def fit_map(ind, x, y):
        """Assigns to each partition the correct values of strengths to allow
        computations in parallel over the pij matrix.

        Note that this is done to ensure that each partition can compute
        pij and pji in the same loop to be able to compute undirected
        properties of the ensemble.

        Parameters
        ----------
        ind: tuple
            a tuple containing the slices of the index of prop_out (ind[0])
            and of prop_in (ind[1]) to iterate over
        x: numpy.ndarray
            the out fitness
        y: numpy.ndarray
            the in fitness

        Output
        ------
        ind: tuple
            as input
        x: tuple
            the relevant slices of x for the iteration over ind
        y: tuple
            the relevant slices of y for the iteration over ind
        """
        il, iu = ind[0]
        jl, ju = ind[1]
        return ind, (x[il:iu], x[jl:ju]), (y[jl:ju], y[il:iu])

    @staticmethod
    def tuple_of_lists_append(x, y):
        for i in range(len(x)):
            x[i].extend(y[i])
        return x

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges(
        p_ij, param, ind_out, ind_in, prop_out, prop_in, prop_dyad, selfloops
    ):
        """Compute the expected number of edges."""
        exp_e = 0.0
        for i in range(ind_out[1] - ind_out[0]):
            p_out_i = prop_out[i]
            for j in range(ind_in[1] - ind_in[0]):
                p_in_j = prop_in[j]
                if (ind_out[0] + i != ind_in[0] + j) | selfloops:
                    exp_e += p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))

        return exp_e

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_degrees(
        p_ij, param, ind_out, ind_in, prop_out, prop_in, prop_dyad, num_v, selfloops
    ):
        """Compute the expected undirected, in and out degree sequences."""
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)

        if ind_out == ind_in:
            fold = True
        else:
            fold = False

        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[0][i]
            p_in_i = prop_in[1][i]
            for j in in_range(i, ind_in, fold):
                ind_j = ind_in[0] + j
                p_out_j = prop_out[1][j]
                p_in_j = prop_in[0][j]
                if ind_i != ind_j:
                    pij = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    pji = p_ij(param, p_out_j, p_in_i, prop_dyad(j, i))
                    p = pij + pji - pij * pji
                    exp_d[ind_i] += p
                    exp_d[ind_j] += p
                    exp_d_out[ind_i] += pij
                    exp_d_out[ind_j] += pji
                    exp_d_in[ind_j] += pij
                    exp_d_in[ind_i] += pji
                elif selfloops:
                    pii = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    exp_d[ind_i] += pii
                    exp_d_out[ind_i] += pii
                    exp_d_in[ind_j] += pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_av_nn_prop(
        p_ij,
        param,
        ind_out,
        ind_in,
        prop_out,
        prop_in,
        prop_dyad,
        prop,
        ndir,
        selfloops,
    ):
        """Compute the expected average nearest neighbour property."""
        av_nn = np.zeros(prop.shape, dtype=np.float64)

        if ind_out == ind_in:
            fold = True
        else:
            fold = False

        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[0][i]
            p_in_i = prop_in[1][i]
            for j in in_range(i, ind_in, fold):
                ind_j = ind_in[0] + j
                p_out_j = prop_out[1][j]
                p_in_j = prop_in[0][j]
                if ind_i != ind_j:
                    pij = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    pji = p_ij(param, p_out_j, p_in_i, prop_dyad(j, i))
                    if ndir == "out":
                        av_nn[ind_i] += pij * prop[ind_j]
                        av_nn[ind_j] += pji * prop[ind_i]
                    elif ndir == "in":
                        av_nn[ind_i] += pji * prop[ind_j]
                        av_nn[ind_j] += pij * prop[ind_i]
                    elif ndir == "out-in":
                        p = pij + pji - pij * pji
                        av_nn[ind_i] += p * prop[ind_j]
                        av_nn[ind_j] += p * prop[ind_i]
                    else:
                        raise ValueError("Direction of neighbourhood not right.")
                elif selfloops:
                    pii = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if ndir == "out":
                        av_nn[ind_i] += pii * prop[ind_i]
                    elif ndir == "in":
                        av_nn[ind_i] += pii * prop[ind_i]
                    elif ndir == "out-in":
                        av_nn[ind_i] += pii * prop[ind_i]
                    else:
                        raise ValueError("Direction of neighbourhood not right.")

        return av_nn

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _likelihood(
        logp,
        log1mp,
        param,
        ind_out,
        ind_in,
        prop_out,
        prop_in,
        prop_dyad,
        adj_i,
        adj_j,
        selfloops,
    ):
        """Compute the binary log likelihood of a graph given the fitted model."""
        like = 0
        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[i]
            n = adj_i[ind_i]
            m = adj_i[ind_i + 1]
            j_list = adj_j[n:m]
            for j in range(ind_in[1] - ind_in[0]):
                ind_j = ind_in[0] + j
                p_in_j = prop_in[j]
                if (ind_i != ind_j) | selfloops:
                    # Check if link exists
                    if ind_j in j_list:
                        tmp = logp(param, p_out_i, p_in_j, prop_dyad(i, j))
                    else:
                        tmp = log1mp(param, p_out_i, p_in_j, prop_dyad(i, j))

                    if isinf(tmp):
                        return tmp
                    like += tmp

        return like

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _confusion_matrix(
        thresholds,
        pij,
        param,
        ind_out,
        ind_in,
        prop_out,
        prop_in,
        prop_dyad,
        adj_i,
        adj_j,
        selfloops,
    ):
        """Compute the binary log likelihood of a graph given the fitted model."""
        tp = np.zeros(thresholds.shape, dtype=np.int64)
        fp = np.zeros(thresholds.shape, dtype=np.int64)
        tn = np.zeros(thresholds.shape, dtype=np.int64)
        fn = np.zeros(thresholds.shape, dtype=np.int64)

        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[i]
            n = adj_i[ind_i]
            m = adj_i[ind_i + 1]
            j_list = adj_j[n:m]

            for j in range(ind_in[1] - ind_in[0]):
                ind_j = ind_in[0] + j
                if (ind_i != ind_j) | selfloops:
                    p_in_j = prop_in[j]
                    p = pij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if ind_j in j_list:
                        for k, th in enumerate(thresholds):
                            if p >= th:
                                tp[k] += 1
                            else:
                                fn[k] += 1
                    else:
                        for k, th in enumerate(thresholds):
                            if p >= th:
                                fp[k] += 1
                            else:
                                tn[k] += 1

        return tp, fp, tn, fn

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _binary_sample(
        p_ij, param, ind_out, ind_in, prop_out, prop_in, prop_dyad, selfloops
    ):
        """Sample from the ensemble."""
        rows = List()
        cols = List()
        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[i]
            for j in range(ind_in[1] - ind_in[0]):
                ind_j = ind_in[0] + j
                p_in_j = prop_in[j]
                if (ind_i != ind_j) | selfloops:
                    p = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if rng.random() < p:
                        rows.append(ind_i)
                        cols.append(ind_j)

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample(
        p_ij,
        param,
        ind_out,
        ind_in,
        prop_out,
        prop_in,
        prop_dyad,
        s_out,
        s_in,
        selfloops,
    ):
        """Sample from the ensemble with weights from the CremB model."""
        s_tot = np.sum(s_out)
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in) / s_tot) < 1e-6, msg
        rows = List()
        cols = List()
        vals = List()
        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            p_out_i = prop_out[i]
            for j in range(ind_in[1] - ind_in[0]):
                ind_j = ind_in[0] + j
                p_in_j = prop_in[j]
                if (ind_i != ind_j) | selfloops:
                    p = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if rng.random() < p:
                        rows.append(ind_i)
                        cols.append(ind_j)
                        vals.append(
                            rng.exponential(s_out[ind_i] * s_in[ind_j] / (s_tot * p))
                        )

        return rows, cols, vals


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

        # First argument must be a SparkContext
        if len(args) > 0:
            if isinstance(args[0], SparkContext):
                self.sc = args[0]
            else:
                raise ValueError("First argument must be a SparkContext.")

            # If an argument is passed then it must be a graph
            if len(args) > 1:
                if isinstance(args[1], graphs.Graph):
                    g = args[1]
                    self.num_vertices = g.num_vertices
                    self.num_edges = g.num_edges()
                    if g.weighted:
                        self.total_weight = g.total_weight()
                else:
                    raise ValueError("Second argument passed must be a Graph.")

            if len(args) > 2:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        else:
            raise ValueError("A SparkContext must be passed as the first argument.")

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "p_blocks",
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

        # Ensure that the number of blocks is a positive integer
        if not hasattr(self, "p_blocks"):
            self.p_blocks = 10
        else:
            try:
                assert self.p_blocks / int(self.p_blocks) == 1
                self.p_blocks = int(self.p_blocks)
            except Exception:
                raise ValueError("Number of parallel blocks must be an integer.")

            if self.p_blocks <= 0:
                raise ValueError("Number of parallel blocks must be a positive number.")

        # Ensure number of blocks is smaller than dimensions of graphs
        # it is in general inefficient too have too few elements per partition
        if self.p_blocks > self.num_vertices / 10:
            if floor(self.num_vertices / 10) < 2:
                self.p_blocks = 2
            else:
                self.p_blocks = int(self.num_vertices / 10)

        # Create two RDDs to parallelize computations
        # The first simply divides the pij matrix in blocks
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(self.p_blocks):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append(((x, y), np.ones(x[1] - x[0]), np.ones(y[1] - y[0])))

        self.p_iter_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # the second has a triangular structure allowing
        # to iterate over pij and pji in the same block
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(i + 1):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append((x, y))

        self.p_sym_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # Assign to each parallel partition the correct fitness values
        fin = np.ones(self.num_vertices)
        fout = np.ones(self.num_vertices)
        fmap = self.fit_map
        self.p_sym_rdd = self.p_sym_rdd.map(lambda x: fmap(x, fout, fin))

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
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

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

        # It is necessary to select the elements or pickling will fail
        av_nn = np.empty(prop.shape, dtype=np.float64)
        if (ndir == "out") or (ndir == "in"):
            av_nn[:] = np.sum(self.param[0] * prop, axis=0)
            if not self.selfloops:
                av_nn -= self.param[0] * prop
        elif ndir == "out-in":
            av_nn[:] = np.sum((2 * self.param[0] - self.param[0] ** 2) * prop, axis=0)
            if not self.selfloops:
                av_nn -= (2 * self.param[0] - self.param[0] ** 2) * prop
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
            adj = sp.csr_matrix(g)
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
        g.adj = sp.csr_matrix(
            (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
        )

        return g

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _binary_sample(param, num_v, selfloops):
        """Sample from the ensemble."""
        rows = List()
        cols = List()
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
        rows = List()
        cols = List()
        vals = List()
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
        rows = List()
        cols = List()
        vals = List()
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
        rows = List()
        cols = List()
        vals = List()
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


class FitnessModel(DiGraphEnsemble):
    """The Fitness model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    sc: Spark Context
        the Spark Context
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
        """Return a FitnessModel for the given graph data.

        The model accepts as arguments either: a DiGraph, in which case the
        strengths are used as fitnesses, or directly the fitness sequences (in
        and out). The model accepts the fitness sequences as numpy arrays.
        """
        # First argument must be a SparkContext
        if len(args) > 0:
            if isinstance(args[0], SparkContext):
                self.sc = args[0]
            else:
                raise ValueError("First argument must be a SparkContext.")

            # If an argument is passed then it must be a graph
            if len(args) > 1:
                if isinstance(args[1], graphs.DiGraph):
                    g = args[1]
                    self.num_vertices = g.num_vertices
                    self.num_edges = g.num_edges()
                    self.prop_out = g.out_strength()
                    self.prop_in = g.in_strength()
                else:
                    raise ValueError("Second argument passed must be a DiGraph.")

            if len(args) > 2:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        else:
            raise ValueError("A SparkContext must be passed as the first argument.")

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "prop_out",
            "prop_in",
            "p_blocks",
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

        # Ensure that the number of blocks is a positive integer
        if not hasattr(self, "p_blocks"):
            self.p_blocks = 10
        else:
            try:
                assert self.p_blocks / int(self.p_blocks) == 1
                self.p_blocks = int(self.p_blocks)
            except Exception:
                raise ValueError("Number of parallel blocks must be an integer.")

            if self.p_blocks <= 0:
                raise ValueError("Number of parallel blocks must be a positive number.")

        # Ensure number of blocks is smaller than dimensions of graphs
        # it is in general inefficient too have too few elements per partition
        if self.p_blocks > self.num_vertices / 10:
            if floor(self.num_vertices / 10) < 2:
                self.p_blocks = 2
            else:
                self.p_blocks = int(self.num_vertices / 10)

        # Create two RDDs to parallelize computations
        # The first simply divides the pij matrix in blocks
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(self.p_blocks):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append(
                    ((x, y), self.prop_out[x[0] : x[1]], self.prop_in[y[0] : y[1]])
                )

        self.p_iter_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # the second has a triangular structure allowing
        # to iterate over pij and pji in the same block
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(i + 1):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append((x, y))

        self.p_sym_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # Assign to each parallel partition the correct fitness values
        fin = self.prop_in
        fout = self.prop_out
        fmap = self.fit_map
        self.p_sym_rdd = self.p_sym_rdd.map(lambda x: fmap(x, fout, fin))

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
                x_l=0,
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
        f_jac = self.exp_edges_f_jac
        p_jac_ij = self.p_jac_ij
        slflp = self.selfloops
        tmp = self.p_iter_rdd.map(
            lambda x: f_jac(p_jac_ij, delta, x[0][0], x[0][1], x[1], x[2], slflp)
        )
        f, jac = tmp.fold((0, 0), lambda x, y: (x[0] + y[0], x[1] + y[1]))

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac(p_jac_ij, param, ind_out, ind_in, prop_out, prop_in, selfloops):
        """Compute the objective function of the density solver and its
        derivative.
        """
        f = 0.0
        jac = 0.0
        for i in range(ind_out[1] - ind_out[0]):
            p_out_i = prop_out[i]
            for j in range(ind_in[1] - ind_in[0]):
                p_in_j = prop_in[j]
                if (ind_out[0] + i != ind_in[0] + j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ij(param, p_out_i, p_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, x_i, y_j):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        if (x_i == 0) or (y_j == 0):
            return 0.0, 0.0

        if d[0] == 0:
            return 0.0, x_i * y_j

        tmp = x_i * y_j
        tmp1 = d[0] * tmp
        if isinf(tmp1):
            return 1.0, 0.0
        else:
            return tmp1 / (1 + tmp1), tmp / (1 + tmp1) ** 2

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j
        if isinf(tmp):
            return 1.0
        else:
            return tmp / (1 + tmp)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, x_i, y_j, z_ij):
        """Compute the log probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return -np.infty

        tmp = d[0] * x_i * y_j
        if isinf(tmp):
            return 0.0
        else:
            return log(tmp / (1 + tmp))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(d, x_i, y_j, z_ij):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j
        if isinf(tmp):
            return -np.infty
        else:
            return log1p(-tmp / (1 + tmp))


class ScaleInvariantModel(FitnessModel):
    """The Scale Invariant model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    sc: Spark Context
        the Spark Context
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
    def p_jac_ij(d, x_i, y_j):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        if (x_i == 0) or (y_j == 0):
            return 0.0, 0.0

        if d[0] == 0:
            return 0.0, x_i * y_j

        tmp = x_i * y_j
        tmp1 = d[0] * tmp
        if isinf(tmp1):
            return 1.0, 0.0
        else:
            return -expm1(-tmp1), tmp * exp(-tmp1)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j
        if isinf(tmp):
            return 1.0
        else:
            return -expm1(-tmp)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, x_i, y_j, z_ij):
        """Compute the log probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return -np.infty

        tmp = d[0] * x_i * y_j
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
        if (x_i == 0) or (y_j == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j
        if isinf(tmp):
            return -np.infty
        else:
            return -tmp


class MultiGraphEnsemble(GraphEnsemble):
    """General class for MultiGraph ensembles."""

    pass


class MultiDiGraphEnsemble(DiGraphEnsemble):
    """General class for MultiDiGraph ensembles.

    All ensembles are assumed to have independent edges whose probabilities
    depend only on a set of parameters (param), a set of node specific out and
    in properties (prop_out and prop_in), and a set of dyadic properties
    (prop_dyad). The ensemble is defined by the probability function
    pij(param, prop_out, prop_in, prop_dyad) and by the labelled edge
    probability pijk(param, prop_out, prop_in, prop_dyad).

    Methods
    -------
    expected_num_edges:
        Compute the expected number of edges in the ensemble.
    expected_num_edges_label:
        Compute the expected number of edges in the ensemble.
    expected_degree:
        Compute the expected undirected degree of each node.
    expected_out_degree:
        Compute the expected out degree of each node.
    expected_in_degree:
        Compute the expected in degree of each node.
    expected_degree_label:
        Compute the expected undirected degree of each node.
    expected_out_degree_label:
        Compute the expected out degree of each node.
    expected_in_degree_label:
        Compute the expected in degree of each node.
    expected_av_nn_property:
        Compute the expected average of the given property of the nearest
        neighbours of each node.
    expected_av_nn_degree:
        Compute the expected average of the degree of the nearest
        neighbours of each node.
    log_likelihood:
        Compute the likelihood of the given graph.
    sample:
        Return a sample from the ensemble.

    """

    def expected_num_edges(self, recompute=False):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges_label") or recompute:
            # It is necessary to select the elements or pickling will fail
            delta = self.param
            slflp = self.selfloops
            p_ij = self.p_ij
            pdyad = self.prop_dyad
            e_fun = self.exp_edges
            tmp = self.p_iter_rdd.map(
                lambda x: e_fun(
                    p_ij,
                    delta,
                    x[0][0],
                    x[0][1],
                    x[1].indptr,
                    x[2].indptr,
                    x[1].indices,
                    x[2].indices,
                    x[1].data,
                    x[2].data,
                    pdyad,
                    slflp,
                )
            )
            self._exp_num_edges = tmp.fold(0, lambda x, y: x + y)

        return self._exp_num_edges

    def expected_num_edges_label(self, recompute=False):
        """Compute the expected number of edges (per label)."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted before hand.")

        if not hasattr(self, "_exp_num_edges_label") or recompute:
            # Initialize each layer
            e_fun = self.exp_edges_layer
            p_ijk = self.p_ijk
            delta = self.param
            pdyad = self.prop_dyad
            slflp = self.selfloops
            tmp_rdd = self.layers_rdd.map(
                lambda x: (
                    x[0],
                    e_fun(
                        p_ijk,
                        delta[x[0]],
                        x[1].indices,
                        x[1].data,
                        x[2].indices,
                        x[2].data,
                        pdyad,
                        slflp,
                    ),
                )
            )

            # Collect and assign results for each layer
            self._exp_num_edges_label = np.zeros(self.num_labels, dtype=np.float64)
            tmp = tmp_rdd.collect()
            for i, v in tmp:
                # Update results and check convergence
                self._exp_num_edges_label[i] = v

        return self._exp_num_edges_label

    def expected_degree(self, recompute=False):
        """Compute the expected undirected/out/in degree."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_degree") or recompute:
            # It is necessary to select the elements or pickling will fail
            delta = self.param
            slflp = self.selfloops
            num_v = self.num_vertices
            p_ij = self.p_ij
            pdyad = self.prop_dyad
            e_fun = self.exp_degrees
            tmp = self.p_sym_rdd.map(
                lambda x: e_fun(
                    p_ij,
                    delta,
                    x[0][0],
                    x[0][1],
                    (x[1][0].indptr, x[1][1].indptr),
                    (x[2][0].indptr, x[2][1].indptr),
                    (x[1][0].indices, x[1][1].indices),
                    (x[2][0].indices, x[2][1].indices),
                    (x[1][0].data, x[1][1].data),
                    (x[2][0].data, x[2][1].data),
                    pdyad,
                    num_v,
                    slflp,
                )
            )

            # Initialize results
            exp_d = np.zeros(num_v, dtype=np.float64)
            exp_d_out = np.zeros(num_v, dtype=np.float64)
            exp_d_in = np.zeros(num_v, dtype=np.float64)
            res = tmp.fold(
                (exp_d, exp_d_out, exp_d_in),
                lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
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

    def expected_degree_by_label(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if not hasattr(self, "_exp_degree_label") or recompute:
            # It is necessary to select the elements or pickling will fail
            delta = self.param
            slflp = self.selfloops
            num_v = self.num_vertices
            p_ijk = self.p_ijk
            pdyad = self.prop_dyad
            e_fun = self.exp_degrees_layer
            app_fun = self.tuple_of_lists_append
            tmp = self.layers_rdd.map(
                lambda x: e_fun(
                    p_ijk,
                    delta[x[0]],
                    x[1].indices,
                    x[2].indices,
                    x[1].data,
                    x[2].data,
                    pdyad,
                    x[0],
                    num_v,
                    slflp,
                )
            )

            # Initialize results
            res = tmp.fold(([], [], []), lambda x, y: app_fun(x, y))
            d = np.array(res[0])
            d_out = np.array(res[1])
            d_in = np.array(res[2])
            self._exp_degree_label = sp.coo_array(
                (d[:, 2], (d[:, 0], d[:, 1])),
                shape=(self.num_vertices, self.num_labels),
            ).tocsr()
            self._exp_out_degree_label = sp.coo_array(
                (d_out[:, 2], (d_out[:, 0], d_out[:, 1])),
                shape=(self.num_vertices, self.num_labels),
            ).tocsr()
            self._exp_in_degree_label = sp.coo_array(
                (d_in[:, 2], (d_in[:, 0], d_in[:, 1])),
                shape=(self.num_vertices, self.num_labels),
            ).tocsr()

        return self._exp_degree_label

    def expected_out_degree_by_label(self, recompute=False):
        """Compute the expected out degree."""
        if not hasattr(self, "_exp_out_degree_label") or recompute:
            _ = self.expected_degree_by_label()

        return self._exp_out_degree_label

    def expected_in_degree_by_label(self, recompute=False):
        """Compute the expected in degree."""
        if not hasattr(self, "_exp_in_degree_label") or recompute:
            _ = self.expected_degree_by_label()

        return self._exp_in_degree_label

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

        # It is necessary to select the elements or pickling will fail
        e_fun = self.exp_av_nn_prop
        p_ij = self.p_ij
        pdyad = self.prop_dyad
        delta = self.param

        tmp = self.p_sym_rdd.map(
            lambda x: e_fun(
                p_ij,
                delta,
                x[0][0],
                x[0][1],
                (x[1][0].indptr, x[1][1].indptr),
                (x[2][0].indptr, x[2][1].indptr),
                (x[1][0].indices, x[1][1].indices),
                (x[2][0].indices, x[2][1].indices),
                (x[1][0].data, x[1][1].data),
                (x[2][0].data, x[2][1].data),
                pdyad,
                prop,
                ndir,
                selfloops,
            )
        )
        av_nn = tmp.fold(np.zeros(prop.shape, dtype=np.float64), lambda x, y: x + y)

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        # Restore model self-loops properties if they have been modified
        if tmp_self != self.selfloops:
            self.selfloops = tmp_self
            if hasattr(self, "_exp_out_degree"):
                del self._exp_out_degree
            if hasattr(self, "_exp_in_degree"):
                del self._exp_in_degree
            if hasattr(self, "_exp_degree"):
                del self._exp_degree

        return av_nn

    def expected_av_nn_degree(
        self,
        ddir="out",
        ndir="out",
        selfloops=False,
        deg_recompute=False,
        recompute=False,
    ):
        """Computes the expected value of the nearest neighbour average of
        the degree.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        # Compute property name
        name = "exp_av_" + ndir.replace("-", "_") + "_nn_d_" + ddir.replace("-", "_")

        if not hasattr(self, name) or recompute or deg_recompute:
            # Set selfloops option
            tmp_self = self.selfloops
            if selfloops is None:
                selfloops = self.selfloops
            elif selfloops != self.selfloops:
                deg_recompute = True
                self.selfloops = selfloops

            # Compute correct expected degree
            if ddir == "out":
                deg = self.expected_out_degree(recompute=deg_recompute)
            elif ddir == "in":
                deg = self.expected_in_degree(recompute=deg_recompute)
            elif ddir == "out-in":
                deg = self.expected_degree(recompute=deg_recompute)
            else:
                raise ValueError("Degree type not recognised.")

            # Compute property and set attribute
            res = self.expected_av_nn_property(
                deg, ndir=ndir, selfloops=selfloops, deg_recompute=False
            )
            setattr(self, name, res)

            # Restore model self-loops properties if they have been modified
            if tmp_self != self.selfloops:
                self.selfloops = tmp_self
                if hasattr(self, "_exp_out_degree"):
                    del self._exp_out_degree
                if hasattr(self, "_exp_in_degree"):
                    del self._exp_in_degree
                if hasattr(self, "_exp_degree"):
                    del self._exp_degree

        return getattr(self, name)

    def log_likelihood(self, g, selfloops=None):
        """Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if selfloops is None:
            selfloops = self.selfloops

        if isinstance(g, graphs.MultiGraph):
            # Extract binary adjacency tensor from graph
            adj = g.adjacency_tensor(directed=True, weighted=False)
            tensor = True
        elif isinstance(g, graphs.Graph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(directed=True, weighted=False)
            tensor = False
        elif sp.issparse(g):
            adj = g.asformat("csr")
            tensor = False
        elif isinstance(g, np.ndarray):
            if g.ndim == 3:
                adj = [sp.csr_matrix(g[i, :, :]) for i in range(g.shape[0])]
                tensor = True
            elif g.ndim == 2:
                adj = sp.csr_matrix(g)
                tensor = False
            else:
                raise ValueError("Adjacency array is neither 2 nor 3 dimensional.")
        elif isinstance(g, list):
            adj = [sp.csr_matrix(x) for x in g]
            tensor = True
        else:
            raise ValueError("g input not a graph or adjacency matrix.")

        # Ensure dimensions are correct
        if tensor:
            msg = (
                "Passed graph adjacency tensor does not have the "
                "correct number of layers: {0} instead of {1}".format(
                    len(adj), self.num_labels
                )
            )
            if len(adj) != self.num_labels:
                raise ValueError(msg)
            for i, x in enumerate(adj):
                if x.shape != (self.num_vertices, self.num_vertices):
                    msg = (
                        "Passed graph adjacency tensor does not have the "
                        "correct shape in layer{2}: {0} instead of {1}".format(
                            adj.shape, (self.num_vertices, self.num_vertices), i
                        )
                    )
                    raise ValueError(msg)
        else:
            if adj.shape != (self.num_vertices, self.num_vertices):
                msg = (
                    "Passed graph adjacency matrix does not have the "
                    "correct shape: {0} instead of {1}".format(
                        adj.shape, (self.num_vertices, self.num_vertices)
                    )
                )
                raise ValueError(msg)

        # Compute log likelihood of graph
        if tensor:
            # Compute log likelihood of graph
            e_fun = self._likelihood_layer
            pdyad = self.prop_dyad
            delta = self.param
            logp = self.logp_ijk
            log1mp = self.log1mp_ijk
            tmp = self.layers_rdd.map(
                lambda x: e_fun(
                    logp,
                    log1mp,
                    delta[x[0]],
                    x[1].indices,
                    x[2].indices,
                    x[1].data,
                    x[2].data,
                    pdyad,
                    adj[x[0]].indptr,
                    adj[x[0]].indices,
                    selfloops,
                )
            )
            like = tmp.fold(0.0, lambda x, y: x + y)

        else:
            # Compute log likelihood of graph
            e_fun = self._likelihood
            pdyad = self.prop_dyad
            delta = self.param
            logp = self.logp
            log1mp = self.log1mp
            tmp = self.p_iter_rdd.map(
                lambda x: e_fun(
                    logp,
                    log1mp,
                    delta,
                    x[0][0],
                    x[0][1],
                    x[1].indptr,
                    x[2].indptr,
                    x[1].indices,
                    x[2].indices,
                    x[1].data,
                    x[2].data,
                    pdyad,
                    adj.indptr,
                    adj.indices,
                    selfloops,
                )
            )
            like = tmp.fold(0.0, lambda x, y: x + y)

        return like

    def confusion_matrix(self, g, thresholds=None, selfloops=None):
        """Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, "param"):
            raise Exception("Ensemble has to be fitted before.")

        if selfloops is None:
            selfloops = self.selfloops

        # If thresholds are not given or is an int compute automatically
        if thresholds is None:
            p_max = -expm1(-self.param[0] * self.prop_out.max() * self.prop_in.max())
            p_min = -expm1(
                -self.param[0]
                * self.prop_out[self.prop_out != 0].min()
                * self.prop_in[self.prop_in != 0].min()
            )
            thresholds = np.logspace(log(p_min), log(p_max), 20)

        elif isinstance(thresholds, int):
            p_max = -expm1(-self.param[0] * self.prop_out.max() * self.prop_in.max())
            p_min = -expm1(
                -self.param[0]
                * self.prop_out[self.prop_out != 0].min()
                * self.prop_in[self.prop_in != 0].min()
            )
            thresholds = np.logspace(log(p_min), log(p_max), thresholds)

        elif not isinstance(thresholds, np.ndarray):
            raise ValueError("Thresholds must be an array or an integer.")

        if isinstance(g, graphs.Graph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(directed=True, weighted=False)
        elif sp.issparse(g):
            adj = g.asformat("csr")
        elif isinstance(g, np.ndarray):
            adj = sp.csr_matrix(g)
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

        # Compute positive and negative rates of graph
        e_fun = self._confusion_matrix
        p = self.p_ij
        delta = self.param
        pdyad = self.prop_dyad
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(
                thresholds,
                p,
                delta,
                x[0][0],
                x[0][1],
                x[1].indptr,
                x[2].indptr,
                x[1].indices,
                x[2].indices,
                x[1].data,
                x[2].data,
                pdyad,
                adj.indptr,
                adj.indices,
                selfloops,
            )
        )
        tp = np.zeros(thresholds.shape, dtype=np.int64)
        fp = np.zeros(thresholds.shape, dtype=np.int64)
        tn = np.zeros(thresholds.shape, dtype=np.int64)
        fn = np.zeros(thresholds.shape, dtype=np.int64)
        res = tmp.fold(
            (tp, fp, tn, fn),
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]),
        )

        return res[0], res[1], res[2], res[3]

    def sample(
        self,
        ref_g=None,
        weights=None,
        out_strength_label=None,
        in_strength_label=None,
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
        g = graphs.MultiDiGraph.__new__(graphs.MultiDiGraph)

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = g.get_num_bytes(g.num_vertices)
        g.id_dtype = np.dtype("u" + str(num_bytes))
        g.num_labels = self.num_labels
        num_bytes = g.get_num_bytes(g.num_labels)
        g.label_dtype = np.dtype("u" + str(num_bytes))

        # Check if reference graph is available
        if ref_g is not None:
            if hasattr(ref_g, "num_groups"):
                g.num_groups = ref_g.num_groups
                g.group_dict = ref_g.group_dict
                g.group_dtype = ref_g.group_dtype
                g.groups = ref_g.groups

            g.id_dict = ref_g.id_dict
            g.label_dict = ref_g.label_dict
        else:
            g.id_dict = {}
            for i in range(g.num_vertices):
                g.id_dict[i] = i
            g.label_dict = {}
            for i in range(g.num_labels):
                g.label_dict[i] = i

        # If weights are given check that they make sense
        if out_strength_label is None:
            s_out_l = self.prop_out.tocsc()
        else:
            # Check dimensions
            if isinstance(out_strength_label, np.ndarray):
                out_strength_label = sp.csc_matrix(out_strength_label)

            if sp.issparse(out_strength_label):
                msg = (
                    "Out strength by label must have shape (num_vertices, "
                    "num_labels)."
                )
                assert out_strength_label.shape == (
                    self.num_vertices,
                    self.num_labels,
                ), msg
                s_out_l = out_strength_label.tocsc()
            else:
                raise ValueError("Out strength by label must be an array.")

        if in_strength_label is None:
            s_in_l = self.prop_in.tocsc()
        else:
            if isinstance(in_strength_label, np.ndarray):
                in_strength_label = sp.csc_matrix(in_strength_label)

            if sp.issparse(in_strength_label):
                msg = (
                    "In strength by label must have shape (num_vertices, "
                    "num_labels)."
                )
                assert in_strength_label.shape == (
                    self.num_vertices,
                    self.num_labels,
                ), msg
                s_in_l = in_strength_label.tocsc()
            else:
                raise ValueError("In strength by label must be an array.")

        # Sample edges by layer
        g.adj = [None] * self.num_labels

        if weights is None:
            e_fun = self._binary_sample_layer
            selfloops = self.selfloops
            p_ijk = self.p_ijk
            pdyad = self.prop_dyad
            delta = self.param
            tmp = self.layers_rdd.map(
                lambda x: (
                    x[0],
                    e_fun(
                        p_ijk,
                        delta[x[0]],
                        x[1].indices,
                        x[2].indices,
                        x[1].data,
                        x[2].data,
                        pdyad,
                        selfloops,
                    ),
                )
            )
            for i, (rows, cols) in tmp.collect():
                # Convert to adjacency matrix
                g.adj[i] = sp.csr_matrix(
                    (np.ones(len(rows), dtype=bool), (rows, cols)),
                    shape=(g.num_vertices, g.num_vertices),
                )

        elif weights == "cremb":
            e_fun = self._cremb_sample_layer
            selfloops = self.selfloops
            p_ijk = self.p_ijk
            pdyad = self.prop_dyad
            delta = self.param
            tmp = self.layers_rdd.map(
                lambda x: (
                    x[0],
                    e_fun(
                        p_ijk,
                        delta[x[0]],
                        x[1].indices,
                        x[2].indices,
                        x[1].data,
                        x[2].data,
                        s_out_l[:, x[0]].indices,
                        s_out_l[:, x[0]].data,
                        s_in_l[:, x[0]].indices,
                        s_in_l[:, x[0]].data,
                        pdyad,
                        selfloops,
                    ),
                )
            )
            for i, (rows, cols, vals) in tmp.collect():
                # Convert to adjacency matrix
                g.adj[i] = sp.csr_matrix(
                    (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
                )

        else:
            raise ValueError("Weights method not recognised or implemented.")

        return g

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges(
        p_ij,
        param,
        ind_out,
        ind_in,
        indptr_out,
        indptr_in,
        lbl_out,
        lbl_in,
        prop_out,
        prop_in,
        pdyad,
        slflp,
    ):
        """Compute the objective function of the density solver and its
        derivative.
        """
        f = 0.0
        for i in range(ind_out[1] - ind_out[0]):
            f_out_i = ind_out[0] + i
            f_out_l = lbl_out[indptr_out[i] : indptr_out[i + 1]]
            f_out_v = prop_out[indptr_out[i] : indptr_out[i + 1]]
            for j in range(ind_in[1] - ind_in[0]):
                f_in_j = ind_in[0] + j
                if (f_out_i != f_in_j) | slflp:
                    f_in_l = lbl_in[indptr_in[j] : indptr_in[j + 1]]
                    f_in_v = prop_in[indptr_in[j] : indptr_in[j + 1]]
                    f += p_ij(param, f_out_l, f_out_v, f_in_l, f_in_v, pdyad)

        return f

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_layer(
        p_ijk, param, ind_out, prop_out, ind_in, prop_in, pdyad, selfloops
    ):
        """Compute the objective function of the layer density solver and its
        derivative.
        """
        f = 0.0
        for i, out_i in enumerate(ind_out):
            for j, in_j in enumerate(ind_in):
                if (out_i != in_j) | selfloops:
                    f += p_ijk(param, prop_out[i], prop_in[j], pdyad(i, j))

        return f

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_degrees(
        p_ij,
        param,
        ind_out,
        ind_in,
        ptr_out,
        ptr_in,
        lbl_out,
        lbl_in,
        val_out,
        val_in,
        pdyad,
        num_v,
        selfloops,
    ):
        """Compute the expected undirected, in and out degree sequences."""
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)

        if ind_out == ind_in:
            fold = True
        else:
            fold = False

        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            l_out_i = lbl_out[0][ptr_out[0][i] : ptr_out[0][i + 1]]
            f_out_i = val_out[0][ptr_out[0][i] : ptr_out[0][i + 1]]
            l_in_i = lbl_in[1][ptr_in[1][i] : ptr_in[1][i + 1]]
            f_in_i = val_in[1][ptr_in[1][i] : ptr_in[1][i + 1]]
            for j in in_range(i, ind_in, fold):
                ind_j = ind_in[0] + j
                l_out_j = lbl_out[1][ptr_out[1][j] : ptr_out[1][j + 1]]
                f_out_j = val_out[1][ptr_out[1][j] : ptr_out[1][j + 1]]
                l_in_j = lbl_in[0][ptr_in[0][j] : ptr_in[0][j + 1]]
                f_in_j = val_in[0][ptr_in[0][j] : ptr_in[0][j + 1]]

                if ind_i != ind_j:
                    pij = p_ij(param, l_out_i, f_out_i, l_in_j, f_in_j, pdyad(i, j))
                    pji = p_ij(param, l_out_j, f_out_j, l_in_i, f_in_i, pdyad(i, j))
                    p = pij + pji - pij * pji
                    exp_d[ind_i] += p
                    exp_d[ind_j] += p
                    exp_d_out[ind_i] += pij
                    exp_d_out[ind_j] += pji
                    exp_d_in[ind_j] += pij
                    exp_d_in[ind_i] += pji
                elif selfloops:
                    pii = p_ij(param, l_out_i, f_out_i, l_in_j, f_in_j, pdyad(i, j))
                    exp_d[ind_i] += pii
                    exp_d_out[ind_i] += pii
                    exp_d_in[ind_j] += pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_degrees_layer(
        p_ijk, param, ind_out, ind_in, prop_out, prop_in, pdyad, lbl, num_v, selfloops
    ):
        """Compute the expected undirected, in and out degree sequences."""
        d = {}
        d_out = {}
        d_in = {}

        for i, ind_i in enumerate(ind_out):
            f_out_i = prop_out[i]
            for j, ind_j in enumerate(ind_in):
                f_in_j = prop_in[j]
                if ind_i != ind_j:
                    pij = p_ijk(param, f_out_i, f_in_j, pdyad(ind_i, ind_j))

                    if (ind_i in ind_in) and (ind_j in ind_out):
                        f_in_i = prop_in[np.where(ind_in == ind_i)][0]
                        f_out_j = prop_out[np.where(ind_out == ind_j)][0]
                        pji = p_ijk(param, f_out_j, f_in_i, pdyad(ind_i, ind_j))
                        p = pij + pji - pij * pji
                    else:
                        pji = 0.0
                        p = pij

                    key = (np.int64(ind_i), np.int64(lbl))
                    if key in d:
                        d[key] += p
                        d_out[key] += pij
                        d_in[key] += pji
                    else:
                        d[key] = p
                        d_out[key] = pij
                        d_in[key] = pji

                    key = (np.int64(ind_j), np.int64(lbl))
                    if key in d:
                        d[key] += p
                        d_out[key] += pji
                        d_in[key] += pij
                    else:
                        d[key] = p
                        d_out[key] = pji
                        d_in[key] = pij

                elif selfloops:
                    pii = p_ijk(param, f_out_i, f_in_j, pdyad(ind_i, ind_j))
                    key = (np.int64(ind_i), np.int64(lbl))
                    if key in d:
                        d[key] += pii
                        d_out[key] += pii
                        d_in[key] += pii
                    else:
                        d[key] = pii
                        d_out[key] = pii
                        d_in[key] = pii

        # Convert dictionary to lists
        exp_d = List()
        exp_d_out = List()
        exp_d_in = List()
        for key in d.keys():
            exp_d.append([key[0], key[1], d[key]])
            exp_d_out.append([key[0], key[1], d_out[key]])
            exp_d_in.append([key[0], key[1], d_in[key]])

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_av_nn_prop(
        p_ij,
        param,
        ind_out,
        ind_in,
        indptr_out,
        indptr_in,
        lbl_out,
        lbl_in,
        prop_out,
        prop_in,
        pdyad,
        prop,
        ndir,
        slflp,
    ):
        """Compute the expected average nearest neighbour property."""
        av_nn = np.zeros(prop.shape, dtype=np.float64)

        if ind_out == ind_in:
            fold = True
        else:
            fold = False

        for i in range(ind_out[1] - ind_out[0]):
            ind_i = ind_out[0] + i
            l_out_i = lbl_out[0][indptr_out[0][i] : indptr_out[0][i + 1]]
            f_out_i = prop_out[0][indptr_out[0][i] : indptr_out[0][i + 1]]
            l_in_i = lbl_in[1][indptr_in[1][i] : indptr_in[1][i + 1]]
            f_in_i = prop_in[1][indptr_in[1][i] : indptr_in[1][i + 1]]
            for j in in_range(i, ind_in, fold):
                ind_j = ind_in[0] + j
                l_out_j = lbl_out[1][indptr_out[1][j] : indptr_out[1][j + 1]]
                f_out_j = prop_out[1][indptr_out[1][j] : indptr_out[1][j + 1]]
                l_in_j = lbl_in[0][indptr_in[0][j] : indptr_in[0][j + 1]]
                f_in_j = prop_in[0][indptr_in[0][j] : indptr_in[0][j + 1]]

                if ind_i != ind_j:
                    pij = p_ij(
                        param, l_out_i, f_out_i, l_in_j, f_in_j, pdyad(ind_i, ind_j)
                    )
                    pji = p_ij(
                        param, l_out_j, f_out_j, l_in_i, f_in_i, pdyad(ind_i, ind_j)
                    )
                    if ndir == "out":
                        av_nn[ind_i] += pij * prop[ind_j]
                        av_nn[ind_j] += pji * prop[ind_i]
                    elif ndir == "in":
                        av_nn[ind_i] += pji * prop[ind_j]
                        av_nn[ind_j] += pij * prop[ind_i]
                    elif ndir == "out-in":
                        p = pij + pji - pij * pji
                        av_nn[ind_i] += p * prop[ind_j]
                        av_nn[ind_j] += p * prop[ind_i]
                    else:
                        raise ValueError("Direction of neighbourhood not right.")
                elif slflp:
                    pii = p_ij(
                        param, l_out_i, f_out_i, l_in_j, f_in_j, pdyad(ind_i, ind_j)
                    )
                    if ndir == "out":
                        av_nn[ind_i] += pii * prop[ind_i]
                    elif ndir == "in":
                        av_nn[ind_i] += pii * prop[ind_i]
                    elif ndir == "out-in":
                        av_nn[ind_i] += pii * prop[ind_i]
                    else:
                        raise ValueError("Direction of neighbourhood not right.")

        return av_nn

    @staticmethod
    # @jit(nopython=True)  # pragma: no cover
    def _likelihood(
        logp,
        log1mp,
        param,
        ind_out,
        ind_in,
        indptr_out,
        indptr_in,
        lbl_out,
        lbl_in,
        prop_out,
        prop_in,
        pdyad,
        adj_i,
        adj_j,
        slflp,
    ):
        """Compute the objective function of the density solver and its
        derivative.
        """
        like = 0.0
        for i in range(ind_out[1] - ind_out[0]):
            f_out_i = ind_out[0] + i
            f_out_l = lbl_out[indptr_out[i] : indptr_out[i + 1]]
            f_out_v = prop_out[indptr_out[i] : indptr_out[i + 1]]
            j_list = adj_j[adj_i[f_out_i] : adj_i[f_out_i + 1]]
            for j in range(ind_in[1] - ind_in[0]):
                f_in_j = ind_in[0] + j
                if (f_out_i != f_in_j) | slflp:
                    f_in_l = lbl_in[indptr_in[j] : indptr_in[j + 1]]
                    f_in_v = prop_in[indptr_in[j] : indptr_in[j + 1]]
                    # Check if link exists
                    if f_in_j in j_list:
                        tmp = logp(param, f_out_l, f_out_v, f_in_l, f_in_v, pdyad(i, j))
                    else:
                        tmp = log1mp(
                            param, f_out_l, f_out_v, f_in_l, f_in_v, pdyad(i, j)
                        )

                    if isinf(tmp):
                        return tmp
                    like += tmp

        return like

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _likelihood_layer(
        logp_ijk,
        log1mp_ijk,
        param,
        ind_out,
        ind_in,
        prop_out,
        prop_in,
        prop_dyad,
        indptr,
        indices,
        selfloops,
    ):
        """Compute the log likelihood for this layer."""
        like = np.float64(0.0)
        N = len(indptr) - 1

        # Iterate over links in adj to check that there are no links with p==0
        for i in range(N):
            n = indptr[i]
            m = indptr[i + 1]
            if n != m:
                ind = ind_out == i
                if not np.any(ind):
                    return -np.infty
                if np.any(prop_out[ind] == 0):
                    return -np.infty

                j_list = indices[n:m]
                for j in j_list:
                    ind = ind_in == j
                    if not np.any(ind):
                        return -np.infty
                    if np.any(prop_in[ind] == 0):
                        return -np.infty

        # Now compute likelihood due to non-zero values of pijk
        for i, out_i in zip(ind_out, prop_out):
            n = indptr[i]
            m = indptr[i + 1]
            j_list = indices[n:m]
            for j, in_j in zip(ind_in, prop_in):
                if (i != j) | selfloops:
                    # Check if link exists
                    if j in j_list:
                        tmp = logp_ijk(param, out_i, in_j, prop_dyad(i, j))
                    else:
                        tmp = log1mp_ijk(param, out_i, in_j, prop_dyad(i, j))

                    if isinf(tmp):
                        return tmp
                    like += tmp
        return like

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _confusion_matrix(
        thresholds,
        pij,
        param,
        ind_out,
        ind_in,
        indptr_out,
        indptr_in,
        lbl_out,
        lbl_in,
        prop_out,
        prop_in,
        prop_dyad,
        adj_i,
        adj_j,
        selfloops,
    ):
        """Compute the binary log likelihood of a graph given the fitted model."""
        tp = np.zeros(thresholds.shape, dtype=np.int64)
        fp = np.zeros(thresholds.shape, dtype=np.int64)
        tn = np.zeros(thresholds.shape, dtype=np.int64)
        fn = np.zeros(thresholds.shape, dtype=np.int64)

        for i in range(ind_out[1] - ind_out[0]):
            f_out_i = ind_out[0] + i
            f_out_l = lbl_out[indptr_out[i] : indptr_out[i + 1]]
            f_out_v = prop_out[indptr_out[i] : indptr_out[i + 1]]
            j_list = adj_j[adj_i[f_out_i] : adj_i[f_out_i + 1]]

            for j in range(ind_in[1] - ind_in[0]):
                f_in_j = ind_in[0] + j
                if (f_out_i != f_in_j) | selfloops:
                    f_in_l = lbl_in[indptr_in[j] : indptr_in[j + 1]]
                    f_in_v = prop_in[indptr_in[j] : indptr_in[j + 1]]
                    p = pij(param, f_out_l, f_out_v, f_in_l, f_in_v, prop_dyad(i, j))
                    if f_in_j in j_list:
                        for k, th in enumerate(thresholds):
                            if p >= th:
                                tp[k] += 1
                            else:
                                fn[k] += 1
                    else:
                        for k, th in enumerate(thresholds):
                            if p >= th:
                                fp[k] += 1
                            else:
                                tn[k] += 1

        return tp, fp, tn, fn

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _binary_sample_layer(
        p_ijk, delta, out_ind, in_ind, out_val, in_val, pdyad, selfloops
    ):
        """Sample from the ensemble."""
        rows = List()
        cols = List()
        for i, out_i in zip(out_ind, out_val):
            for j, in_j in zip(in_ind, in_val):
                if (i != j) | selfloops:
                    p = p_ijk(delta, out_i, in_j, pdyad(i, j))
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)

        rows_arr = np.empty(len(rows), dtype=np.int64)
        cols_arr = np.empty(len(cols), dtype=np.int64)

        for i, (r, c) in enumerate(zip(rows, cols)):
            rows_arr[i] = r
            cols_arr[i] = c

        return rows_arr, cols_arr

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample_layer(
        p_ijk,
        delta,
        out_ind,
        in_ind,
        out_val,
        in_val,
        s_out_ind,
        s_out_val,
        s_in_ind,
        s_in_val,
        pdyad,
        selfloops,
    ):
        """Sample from the ensemble."""
        s_tot = np.sum(s_out_val)
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in_val) / s_tot) < 1e-6, msg

        rows = List()
        cols = List()
        vals = List()
        for i, out_i in zip(out_ind, out_val):
            for j, in_j in zip(in_ind, in_val):
                if (i != j) | selfloops:
                    p = p_ijk(delta, out_i, in_j, pdyad(i, j))
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        ind_out = s_out_ind == i
                        ind_in = s_in_ind == j
                        if np.any(ind_out) or np.any(ind_in):
                            vals.append(
                                rng.exponential(
                                    s_out_val[ind_out][0]
                                    * s_in_val[ind_in][0]
                                    / (s_tot * p)
                                )
                            )
                        else:
                            vals.append(0.0)

        rows_arr = np.empty(len(rows), dtype=np.int64)
        cols_arr = np.empty(len(cols), dtype=np.int64)
        vals_arr = np.empty(len(vals), dtype=np.float64)

        for i, (r, c, v) in enumerate(zip(rows, cols, vals)):
            rows_arr[i] = r
            cols_arr[i] = c
            vals_arr[i] = v

        return rows_arr, cols_arr, vals_arr


class MultiFitnessModel(MultiDiGraphEnsemble):
    """A generalized fitness model that allows for fitnesses by label.

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

    def __init__(self, *args, **kwargs):
        """Return a MultiFitnessModel for the given graph data.

        The model accepts as arguments either: a MultiDiGraph, in which case
        the strengths per label are used as fitnesses, or directly the fitness
        sequences (in and out). The model accepts the fitness sequences as
        numpy arrays or scipy.sparse arrays (this is the recommended format).
        """
        super().__init__(*args, **kwargs)

        # First argument must be a SparkContext
        if len(args) > 0:
            if isinstance(args[0], SparkContext):
                self.sc = args[0]
            else:
                raise ValueError("First argument must be a SparkContext.")

            # If an argument is passed then it must be a graph
            if len(args) > 1:
                if isinstance(args[1], graphs.MultiDiGraph):
                    g = args[1]
                    self.num_vertices = g.num_vertices
                    self.num_labels = g.num_labels
                    self.num_edges = g.num_edges()
                    self.num_edges_label = g.num_edges_label()
                    self.prop_out = g.out_strength_by_label()
                    self.prop_in = g.in_strength_by_label()
                    self.per_label = True
                else:
                    raise ValueError("Second argument passed must be a MultiDiGraph.")

            if len(args) > 2:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        else:
            raise ValueError("A SparkContext must be passed as the first argument.")

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "num_edges_label",
            "num_labels",
            "prop_out",
            "prop_in",
            "p_blocks",
            "param",
            "selfloops",
            "per_label",
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

        if not hasattr(self, "num_labels"):
            raise ValueError("Number of labels not set.")
        else:
            try:
                assert self.num_labels / int(self.num_labels) == 1
                self.num_labels = int(self.num_labels)
            except Exception:
                raise ValueError("Number of labels must be an integer.")

            if self.num_labels <= 0:
                raise ValueError("Number of labels must be a positive number.")

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

        # Ensure that fitnesses passed adhere to format
        msg = (
            "Node out properties must be a two dimensional array with "
            "shape (num_vertices, num_labels)."
        )
        assert isinstance(self.prop_out, np.ndarray) or sp.issparse(self.prop_out), msg
        assert self.prop_out.shape == (self.num_vertices, self.num_labels), msg

        msg = (
            "Node in properties must be a two dimensional array with "
            "shape (num_vertices, num_labels)."
        )
        assert isinstance(self.prop_in, np.ndarray) or sp.issparse(self.prop_in), msg
        assert self.prop_in.shape == (self.num_vertices, self.num_labels), msg

        # Convert to csr matrices
        self.prop_out = sp.csr_matrix(self.prop_out)
        self.prop_in = sp.csr_matrix(self.prop_in)

        # Ensure that all fitness are positive
        if np.any(self.prop_out.data < 0):
            raise ValueError("Node out properties must contain positive values only.")

        if np.any(self.prop_in.data < 0):
            raise ValueError("Node in properties must contain positive values only.")

        # If num_edges is an array check if it has a single value or
        # one for each label.
        if hasattr(self, "num_edges"):
            if isinstance(self.num_edges, np.ndarray):
                if len(self.num_edges) == self.num_labels:
                    if not hasattr(self, "num_edges_label"):
                        self.num_edges_label = self.num_edges
                    else:
                        raise ValueError(
                            "num_edges is an array of size num_labels"
                            " but num_edges_label is already set."
                        )
                elif len(self.num_edges) == 1:
                    self.num_edges = self.num_edges[0]
                else:
                    raise ValueError(
                        "num_edges must be an array of size one or "
                        "num_labels, not {}.".format(len(self.num_edges))
                    )

        # Ensure that number of constraint matches number of labels or is a
        # single number
        if hasattr(self, "num_edges_label"):
            self.per_label = True
            if not (len(self.num_edges_label) == self.num_labels):
                raise ValueError(
                    "The length of the num_edges_label array does not match "
                    "the number of labels."
                )

            if not np.issubdtype(self.num_edges_label.dtype, np.number):
                raise ValueError("Number of edges per label must be numeric.")

            if np.any(self.num_edges_label < 0):
                raise ValueError("Number of edges per label must be positive.")

        elif hasattr(self, "num_edges"):
            self.per_label = False
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

        # Ensure that number of parameter is a single positive number or it
        # matches the number of labels
        if hasattr(self, "param"):
            self.per_label = True
            if not isinstance(self.param, np.ndarray):
                try:
                    self.param = np.array([p for p in self.param])
                except Exception:
                    self.param = np.array([self.param])
            if len(self.param) == 1:
                self.per_label = False
                self.param = np.array([self.param[0]] * self.num_labels)

            elif not (len(self.param) == self.num_labels):
                raise ValueError("The model requires one or num_labels parameter.")

            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError("Parameters must be numeric.")

            if np.any(self.param < 0):
                raise ValueError("Parameters must be positive in value.")

            # Ensure num edges is a float64
            self.param = self.param.astype(np.float64)

        if not (
            hasattr(self, "num_edges")
            or hasattr(self, "num_edges_label")
            or hasattr(self, "param")
        ):
            raise ValueError("Either num_edges(_label) or param must be set.")

        if "per_label" in kwargs:
            self.per_label = kwargs["per_label"]

        # Ensure that the number of blocks is a positive integer
        if not hasattr(self, "p_blocks"):
            self.p_blocks = 10
        else:
            try:
                assert self.p_blocks / int(self.p_blocks) == 1
                self.p_blocks = int(self.p_blocks)
            except Exception:
                raise ValueError("Number of parallel blocks must be an integer.")

            if self.p_blocks <= 0:
                raise ValueError("Number of parallel blocks must be a positive number.")

        # Ensure number of blocks is smaller than dimensions of graphs
        # it is in general inefficient too have too few elements per partition
        if self.p_blocks > self.num_vertices / 10:
            if floor(self.num_vertices / 10) < 2:
                self.p_blocks = 2
            else:
                self.p_blocks = int(self.num_vertices / 10)

        # Create three RDDs to parallelize computations
        # The first simply divides the pij matrix in blocks
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(self.p_blocks):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append(
                    ((x, y), self.prop_out[x[0] : x[1]], self.prop_in[y[0] : y[1]])
                )

        self.p_iter_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # the second has a triangular structure allowing
        # to iterate over pij and pji in the same block
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i * step, self.num_vertices)
            else:
                x = (i * step, (i + 1) * step)
            for j in range(i + 1):
                if j == self.p_blocks - 1:
                    y = (j * step, self.num_vertices)
                else:
                    y = (j * step, (j + 1) * step)
                elements.append((x, y))

        self.p_sym_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

        # Assign to each parallel partition the correct fitness values
        fin = self.prop_in
        fout = self.prop_out
        fmap = self.fit_map
        self.p_sym_rdd = self.p_sym_rdd.map(lambda x: fmap(x, fout, fin))

        # the third divides the pij in label layers
        # Transform properties to csc format to select labels quickly
        prop_out = self.prop_out.tocsc()
        prop_in = self.prop_in.tocsc()
        elements = [(i, prop_out[:, i], prop_in[:, i]) for i in range(self.num_labels)]
        self.layers_rdd = self.sc.parallelize(elements, numSlices=len(elements)).cache()

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
            Maximum number of iterations.
        verbose: boolean
            If true print debug info while iterating.
        """
        if x0 is None:
            x0 = np.zeros(self.num_labels, dtype=np.float64)

        if not isinstance(x0, np.ndarray):
            try:
                x0 = np.array([p for p in x0])
            except Exception:
                x0 = np.array([x0] * self.num_labels)

        if not (len(x0) == self.num_labels):
            raise ValueError("The model requires one or num_labels initial conditions.")

        if not np.issubdtype(x0.dtype, np.number):
            raise ValueError("x0 must be numeric.")

        if np.any(x0 < 0):
            raise ValueError("x0 must be positive.")

        if method == "density":
            if self.per_label:
                # Ensure that num_edges is set
                if not hasattr(self, "num_edges_label"):
                    raise ValueError(
                        "Number of edges per label must be set for density "
                        "solver with per_label option."
                    )

                self.solver_output = [None] * self.num_labels
                self.param = x0.copy()

                # Initialize each layer with solver function
                d_fit = self.density_fit_layer
                f_jac = self.exp_edges_f_jac_layer
                p_jac = self.p_jac_ijk
                num_e = self.num_edges_label
                slflp = self.selfloops
                sol_rdd = self.layers_rdd.map(
                    lambda x: d_fit(x, x0, f_jac, p_jac, slflp)
                )

                # Map to solver
                sol_rdd = sol_rdd.map(
                    lambda x: (
                        x[0],
                        monotonic_newton_solver(
                            x[1],
                            x[2],
                            num_e[x[0]],
                            atol=atol,
                            rtol=rtol,
                            x_l=0.0,
                            x_u=np.infty,
                            max_iter=maxiter,
                            full_return=True,
                            verbose=verbose,
                        ),
                    )
                )

                # Collect solution to array
                self.param = x0.copy()
                self.solver_output = [None] * self.num_labels
                tmp = sol_rdd.collect()
                for i, sol in tmp:
                    # Update results and check convergence
                    self.param[i] = sol.x[0]
                    self.solver_output[i] = sol
                    if not sol.converged:
                        msg = "Fit of layer {}, did not converge.".format(i)
                        warnings.warn(msg, UserWarning)

            else:
                # Ensure that num_edges is set
                if not hasattr(self, "num_edges"):
                    raise ValueError("Number of edges must be set for density solver.")

                sol = monotonic_newton_solver(
                    x0[0:1],
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

                # Update results and check convergence
                self.param = np.array([sol.x[0]] * self.num_labels)
                self.solver_output = sol

                if not self.solver_output.converged:
                    warnings.warn("Fit did not converge", UserWarning)

        elif method == "mle":
            raise ValueError("Method not implemented.")

        else:
            raise ValueError("The selected method is not valid.")

    def density_fit_fun(self, delta):
        """Return the objective function value and the Jacobian
        for a given value of delta when there are multiple labels per
        edge allowed.
        """
        f_jac = self.exp_edges_f_jac
        p_jac_ij = self.p_jac_ij
        slflp = self.selfloops
        tmp = self.p_iter_rdd.map(
            lambda x: f_jac(
                p_jac_ij,
                delta,
                x[0][0],
                x[0][1],
                x[1].indptr,
                x[2].indptr,
                x[1].indices,
                x[2].indices,
                x[1].data,
                x[2].data,
                slflp,
            )
        )
        f, jac = tmp.fold((0, 0), lambda x, y: (x[0] + y[0], x[1] + y[1]))

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac(
        p_jac_ij,
        param,
        ind_out,
        ind_in,
        indptr_out,
        indptr_in,
        lbl_out,
        lbl_in,
        prop_out,
        prop_in,
        slflp,
    ):
        """Compute the objective function of the density solver and its
        derivative.
        """
        f = np.float64(0.0)
        jac = np.float64(0.0)
        for i in range(ind_out[1] - ind_out[0]):
            f_out_i = ind_out[0] + i
            f_out_l = lbl_out[indptr_out[i] : indptr_out[i + 1]]
            f_out_v = prop_out[indptr_out[i] : indptr_out[i + 1]]
            for j in range(ind_in[1] - ind_in[0]):
                f_in_j = ind_in[0] + j
                if (f_out_i != f_in_j) | slflp:
                    f_in_l = lbl_in[indptr_in[j] : indptr_in[j + 1]]
                    f_in_v = prop_in[indptr_in[j] : indptr_in[j + 1]]
                    p_tmp, jac_tmp = p_jac_ij(param, f_out_l, f_out_v, f_in_l, f_in_v)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    def density_fit_layer(x, x0, f_jac, p_jac, slflp):
        lbl_id = x[0]
        return (
            lbl_id,
            x0[lbl_id : lbl_id + 1],
            lambda y: f_jac(
                p_jac, y[0], x[1].indices, x[1].data, x[2].indices, x[2].data, slflp
            ),
        )

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac_layer(
        p_jac_ijk, param, out_ind, out_val, in_ind, in_val, selfloops
    ):
        """Compute the objective function of the density solver and its
        derivative.
        """
        f = np.float64(0.0)
        jac = np.float64(0.0)
        for i, p_out_i in zip(out_ind, out_val):
            for j, p_in_j in zip(in_ind, in_val):
                if (i != j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ijk(param, p_out_i, p_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, x_lbl, x_val, y_lbl, y_val):
        """Compute the probability of connection between node i and j.

        param is expected to be an array with num_labels elements. All
        properties must be a tuple (indices, values) from a sparse matrix.
        """
        # Initialize result
        i = 0
        j = 0
        val = np.float64(1.0)
        jac = np.float64(0.0)

        # Loop over all possibilities
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (x_val[i] != 0) and (y_val[j] != 0):
                    if d[0] == 0:
                        # In this case pij is zero but jac is not
                        jac += x_val[i] * y_val[j]
                    else:
                        # In this case the pij is not zero
                        tmp = x_val[i] * y_val[j]
                        tmp1 = d[0] * tmp
                        if isinf(tmp1):
                            return 1.0, 0.0
                        else:
                            val /= 1 + tmp1
                            jac += tmp / (1 + tmp1)
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        return 1 - val, val * jac

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
            return tmp1 / (1 + tmp1), tmp / (1 + tmp1) ** 2

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_ij(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the probability of connection between node i and j.

        param is expected to be an array with num_labels elements. All
        properties must be a tuple (indices, values) from a sparse matrix.
        """
        # Initialize result
        i = 0
        j = 0
        val = 1.0

        # Loop over all possibilities
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return 1.0
                    else:
                        val /= 1 + tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        return 1 - val

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the log probability of connection between node i and j."""
        # Initialize result
        i = 0
        j = 0
        val = 1.0

        # Loop over all possibilities
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return 0.0
                    else:
                        val /= 1 + tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        if val == 1.0:
            return -np.infty
        else:
            return log1p(-val)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        # Initialize result
        i = 0
        j = 0
        val = 1.0

        # Loop over all possibilities
        while i < len(x_lbl) and j < len(y_lbl):
            if x_lbl[i] == y_lbl[j]:
                if (d[x_lbl[i]] != 0) and (x_val[i] != 0) and (y_val[j] != 0):
                    tmp = d[x_lbl[i]] * x_val[i] * y_val[j]
                    if isinf(tmp):
                        return -np.infty
                    else:
                        val /= 1 + tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        if val == 1.0:
            return 0.0
        else:
            return log(val)

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
            return tmp / (1 + tmp)

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
            return log(tmp / (1 + tmp))

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp_ijk(d, x_i, y_j, z_ij):
        """Compute the probability of connection between node i and j on
        layer k.
        """
        if (x_i == 0) or (y_j == 0) or (d == 0):
            return 0.0

        tmp = d * x_i * y_j
        if isinf(tmp):
            return -np.infty
        else:
            return log1p(-tmp / (1 + tmp))


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
    def p_jac_ij(d, x_lbl, x_val, y_lbl, y_val):
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
    def p_ij(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the probability of connection between node i and j.

        param is expected to be an array with num_labels elements. All
        properties must be a tuple (indices, values) from a sparse matrix.
        """
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
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
    def logp(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the log probability of connection between node i and j."""
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
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
    def log1mp(d, x_lbl, x_val, y_lbl, y_val, prop_dyad):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        # Initialize result
        i = 0
        j = 0
        val = 0.0

        # Loop over all possibilities
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
