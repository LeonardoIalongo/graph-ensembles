""" This module defines the classes that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """
from . import graphs
from ..solver import monotonic_newton_solver
import numpy as np
import numpy.random as rng
import scipy.sparse as sp
from numba import jit
from math import floor
from math import exp
from math import expm1
from math import log
from math import isinf
import warnings


class GraphEnsemble():
    """ General class for graph ensembles.

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
    pass


class FitnessModel(GraphEnsemble):
    """ The Fitness model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    fit_out: np.ndarray
        the out fitness sequence
    fit_in: np.ndarray
        the in fitness sequence
    num_edges: int
        the total number of edges
    num_vertices: int
        the total number of nodes
    param: float
        the free parameters of the model
    selfloops: bool
        selects if self loops (connections from i to i) are allowed
    """

    def __init__(self, *args, **kwargs):
        """ Return a FitnessModel for the given graph data.
        The model accepts as arguments either: a DiGraph,
        in which case the strengths are used as fitnesses, or
        directly the fitness sequences (in and out).
        The model accepts the fitness sequences as numpy arrays.
        """
        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.DiGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges()
                self.fit_out = g.out_strength()
                self.fit_in = g.in_strength()
            else:
                raise ValueError('First argument passed must be a '
                                 'DiGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'fit_out',
                             'fit_in', 'param', 'selfloops']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)
            else:
                setattr(self, name, kwargs[name])

        # Ensure that all necessary fields have been set
        if not hasattr(self, 'num_vertices'):
            raise ValueError('Number of vertices not set.')
        else:
            try: 
                assert self.num_vertices / int(self.num_vertices) == 1
                self.num_vertices = int(self.num_vertices)
            except Exception:
                raise ValueError('Number of vertices must be an integer.')

            if self.num_vertices <= 0:
                raise ValueError(
                    'Number of vertices must be a positive number.')

        if not hasattr(self, 'fit_out'):
            raise ValueError('fit_out not set.')

        if not hasattr(self, 'fit_in'):
            raise ValueError('fit_in not set.')

        if not hasattr(self, 'selfloops'):
            self.selfloops = False

        # Ensure that fitnesses passed adhere to format (ndarray)
        msg = ("Out fitness must be a numpy array of length " +
               str(self.num_vertices))
        assert isinstance(self.fit_out, np.ndarray), msg
        assert self.fit_out.shape == (self.num_vertices,), msg

        msg = ("In fitness must be a numpy array of length " +
               str(self.num_vertices))
        assert isinstance(self.fit_in, np.ndarray), msg
        assert self.fit_in.shape == (self.num_vertices,), msg

        # Ensure that fitnesses have positive values only
        msg = "Out fitness must contain positive values only."
        assert np.all(self.fit_out >= 0), msg

        msg = "In fitness must contain positive values only."
        assert np.all(self.fit_in >= 0), msg

        # Ensure that number of edges is a positive number
        if hasattr(self, 'num_edges'):
            try: 
                tmp = len(self.num_edges)
                if tmp == 1:
                    self.num_edges = self.num_edges[0]
                else:
                    raise ValueError('Number of edges must be a number.')
            except TypeError:
                pass        
                
            try:
                self.num_edges = self.num_edges * 1.0
            except TypeError:
                raise ValueError('Number of edges must be a number.')

            if self.num_edges < 0:
                raise ValueError(
                    'Number of edges must be a positive number.')
        
        # Ensure that parameter is a single positive number
        if hasattr(self, 'param'):
            if not isinstance(self.param, np.ndarray):
                self.param = np.array([self.param])

            else:
                if not (len(self.param) == 1):
                    raise ValueError(
                        'The FitnessModel requires one parameter.')
            
            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError('Parameters must be numeric.')

            if np.any(self.param < 0):
                raise ValueError('Parameters must be positive.')

        if not (hasattr(self, 'num_edges') or hasattr(self, 'param')):
            raise ValueError('Either num_edges or param must be set.')

    def fit(self, x0=None, method='density', atol=1e-9, 
            xtol=1e-9, maxiter=100, verbose=False):
        """ Fit the parameter either to match the given number of edges or
            using maximum likelihood estimation.

        Parameters
        ----------
        x0: float
            optional initial conditions for parameters
        method: 'density' or 'mle'
            selects whether to fit param using maximum likelihood estimation
            or by ensuring that the expected density matches the given one
        solver: 'Newton-CG' or any scipy minimization solvers
            selects which scipy solver is used for the mle method
        atol : float
            absolute tolerance for the exit condition
        xtol : float
            relative tolerance for the exit condition on consecutive x values
        max_iter : int or float
            maximum number of iteration
        verbose: boolean
            if true print debug info while iterating
        """
        if x0 is None:
            x0 = np.array([0], dtype=np.float64)

        if not isinstance(x0, np.ndarray):
            x0 = np.array([x0])

        if not (len(x0) == 1):
            raise ValueError(
                'The FitnessModel requires one parameter.')

        if not np.issubdtype(x0.dtype, np.number):
            raise ValueError('x0 must be numeric.')

        if np.any(x0 < 0):
            raise ValueError('x0 must be positive.')

        if method == 'density':
            # Ensure that num_edges is set
            if not hasattr(self, 'num_edges'):
                raise ValueError(
                    'Number of edges must be set for density solver.')
            sol = monotonic_newton_solver(
                x0, self.density_fit_fun, atol=atol, xtol=xtol, x_l=0, 
                x_u=np.infty, max_iter=maxiter, full_return=True, 
                verbose=verbose)

        elif method == 'mle':
            raise ValueError("Method not implemented.")

        else:
            raise ValueError("The selected method is not valid.")

        # Update results and check convergence
        self.param = sol.x
        self.solver_output = sol

        if not self.solver_output.converged:
            warnings.warn('Fit did not converge', UserWarning)

    def expected_num_edges(self, recompute=False):
        """ Compute the expected number of edges.
        """
        if not hasattr(self, 'param'):
            raise Exception('Model must be fitted beforehand.')
        
        if not hasattr(self, '_exp_num_edges') or recompute:
            self._exp_num_edges = self.exp_edges(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                self.selfloops)

        return self._exp_num_edges
        
    def expected_degree(self, recompute=False):
        """ Compute the expected undirected degree.
        """
        if not hasattr(self, '_exp_degree') or recompute:
            res = self.exp_degrees(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                self.num_vertices, self.selfloops)
            self._exp_degree = res[0]
            self._exp_out_degree = res[1]
            self._exp_in_degree = res[2]

        return self._exp_degree

    def expected_out_degree(self, recompute=False):
        """ Compute the expected out degree.
        """
        if not hasattr(self, '_exp_out_degree') or recompute:
            res = self.exp_degrees(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                self.num_vertices, self.selfloops)
            self._exp_degree = res[0]
            self._exp_out_degree = res[1]
            self._exp_in_degree = res[2]

        return self._exp_out_degree

    def expected_in_degree(self, recompute=False):
        """ Compute the expected in degree.
        """
        if not hasattr(self, '_exp_in_degree') or recompute:
            res = self.exp_degrees(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                self.num_vertices, self.selfloops)
            self._exp_degree = res[0]
            self._exp_out_degree = res[1]
            self._exp_in_degree = res[2]
        
        return self._exp_in_degree

    def expected_av_nn_property(self, prop, ndir='out', selfloops=False, 
                                deg_recompute=False):
        """ Computes the expected value of the nearest neighbour average of
        the property array. The array must have the first dimension
        corresponding to the vertex index.
        """
        # Check first dimension of property array is correct
        if not prop.shape[0] == self.num_vertices:
            msg = ('Property array must have first dimension size be equal to'
                   ' the number of vertices.')
            raise ValueError(msg)

        # Compute correct expected degree
        if ndir == 'out':
            deg = self.exp_out_degree(recompute=deg_recompute)
        elif ndir == 'in':
            deg = self.exp_in_degree(recompute=deg_recompute)
        elif ndir == 'out-in':
            deg = self.exp_degree(recompute=deg_recompute)
        else:
            raise ValueError('Neighbourhood direction not recognised.')

        # It is necessary to select the elements or pickling will fail
        av_nn = self.exp_av_nn_prop(
            self.p_ij, self.param, self.fit_out, self.fit_in, prop, ndir, 
            self.selfloops)
        
        # Test that mask is the same
        ind = deg != 0
        msg = 'Got a av_nn for an empty neighbourhood.'
        assert np.all(av_nn[~ind] == 0), msg
        
        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        return av_nn

    def expected_av_nn_degree(self, ddir='out', ndir='out', selfloops=False,
                              deg_recompute=False, recompute=False):
        """ Computes the expected value of the nearest neighbour average of
        the degree.
        """
        # Compute property name
        name = ('exp_av_' + ndir.replace('-', '_') + 
                '_nn_d_' + ddir.replace('-', '_'))

        if not hasattr(self, name) or recompute:
            # Compute correct expected degree
            if ndir == 'out':
                deg = self.exp_out_degree(recompute=deg_recompute)
            elif ndir == 'in':
                deg = self.exp_in_degree(recompute=deg_recompute)
            elif ndir == 'out-in':
                deg = self.exp_degree(recompute=deg_recompute)
            else:
                raise ValueError('Degree type not recognised.')

            # Compute property and set attribute
            name = ('exp_av_' + ndir.replace('-', '_') + 
                    '_nn_d_' + ddir.replace('-', '_'))
            res = self.expected_av_nn_property(
                deg, ndir=ndir, selfloops=selfloops, deg_recompute=False)
            setattr(self, name, res)

        return getattr(self, name)

    def log_likelihood(self, g, selfloops=None):
        """ Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before.')

        if selfloops is None:
            selfloops = self.selfloops

        if isinstance(g, graphs.Graph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(directed=True, weighted=False)
        elif isinstance(g, sp.spmatrix):
            adj = g.asformat('csr')
        elif isinstance(g, np.ndarray):
            adj = sp.csr_matrix(g)
        else:
            raise ValueError('g input not a graph or adjacency matrix.')

        # Ensure dimensions are correct
        if adj.shape != (self.num_vertices, self.num_vertices):
            msg = ('Passed graph adjacency matrix does not have the correct '
                   'shape: {0} instead of {1}'.format(
                    adj.shape, (self.num_vertices, self.num_vertices)))
            raise ValueError(msg)

        # Compute log likelihood of graph
        like = self._likelihood(
            self.p_ij, self.param, self.fit_out, self.fit_in, 
            adj.indptr, adj.indices, self.selfloops)

        return like

    def sample(self, ref_g=None, weights=None, out_strength=None, 
               in_strength=None, selfloops=None):
        """ Return a Graph sampled from the ensemble.

        If a reference graph is passed (ref_g) then the properties of the graph
        will be copied to the new samples.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before sampling.')

        if selfloops is None:
            selfloops = self.selfloops

        # Generate uninitialised graph object
        g = graphs.DiGraph.__new__(graphs.DiGraph)

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = g.get_num_bytes(g.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))

        # Check if reference graph is available
        if ref_g is not None:
            if hasattr(ref_g, 'num_groups'):
                g.num_groups = ref_g.num_groups
                g.group_dict = ref_g.group_dict
                g.group_dtype = ref_g.group_dtype
                g.groups = ref_g.groups

            g.id_dict = ref_g.id_dict
        else:
            g.id_dict = {}
            for x in g.v.id:
                g.id_dict[x] = x

        # Sample edges
        if weights is None:
            rows, cols = self._binary_sample(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                self.selfloops)
            vals = np.ones(len(rows), dtype=bool)
        elif weights == 'cremb':
            if out_strength is None:
                s_out = self.fit_out
            if in_strength is None:
                s_in = self.fit_in
            rows, cols, vals = self._cremb_sample(
                self.p_ij, self.param, self.fit_out, self.fit_in, 
                s_out, s_in, self.selfloops)
        else:
            raise ValueError('Weights method not recognised or implemented.')

        # Convert to adjacency matrix
        g.adj = sp.csr_matrix((vals, (rows, cols)), 
                              shape=(g.num_vertices, g.num_vertices))

        return g

    def density_fit_fun(self, delta):
        """ Return the objective function value and the Jacobian
            for a given value of delta.
        """
        f, jac = self.exp_edges_f_jac(
            self.p_jac_ij, delta, self.fit_out, self.fit_in, self.selfloops)
        f -= self.num_edges
        return f, jac

    @staticmethod              
    @jit(nopython=True)
    def exp_edges_f_jac(p_jac_ij, param, fit_out, fit_in, selfloops):
        """ Compute the objective function of the density solver and its
        derivative.
        """
        f = 0.0
        jac = 0.0
        for i in range(len(fit_out)):
            f_out_i = fit_out[i]
            for j in range(len(fit_in)):
                f_in_j = fit_in[j]
                if (i != j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ij(param[0], f_out_i, f_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)
    def p_ij(d, x_i, y_j):
        """ Compute the probability of connection between node i and j.
        """
        tmp = d*x_i*y_j
        if isinf(tmp):
            return 1.0
        else:
            return tmp / (1 + tmp)

    @staticmethod
    @jit(nopython=True)
    def p_jac_ij(d, x_i, y_j):
        """ Compute the probability of connection and the jacobian 
            contribution of node i and j.
        """
        tmp = x_i*y_j
        tmp1 = d*tmp
        if isinf(tmp1):
            return 1.0, 0.0
        else:
            return tmp1 / (1 + tmp1), tmp / (1 + tmp1)**2

    @staticmethod
    @jit(nopython=True)
    def exp_edges(p_ij, param, fit_out, fit_in, selfloops):
        """ Compute the expected number of edges.
        """
        exp_e = 0.0
        for i in range(len(fit_out)):
            f_out_i = fit_out[i]
            for j in range(len(fit_in)):
                f_in_j = fit_in[j]
                if (i != j) | selfloops:
                    exp_e += p_ij(param[0], f_out_i, f_in_j)

        return exp_e

    @staticmethod
    @jit(nopython=True)
    def exp_degrees(p_ij, param, fit_out, fit_in, num_v, selfloops):
        """ Compute the expected undirected, in and out degree sequences.
        """
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)

        for i in range(len(fit_out)):
            f_out_i = fit_out[i]
            f_in_i = fit_in[i]
            for j in range(i + 1):
                f_out_j = fit_out[j]
                f_in_j = fit_in[j]
                if i != j:
                    pij = p_ij(param[0], f_out_i, f_in_j)
                    pji = p_ij(param[0], f_out_j, f_in_i)
                    p = pij + pji - pij*pji
                    exp_d[i] += p
                    exp_d[j] += p
                    exp_d_out[i] += pij
                    exp_d_out[j] += pji
                    exp_d_in[j] += pij
                    exp_d_in[i] += pji
                elif selfloops:
                    pii = p_ij(param[0], f_out_i, f_in_j)
                    exp_d[i] += pii
                    exp_d_out[i] += pii
                    exp_d_in[j] += pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)
    def exp_av_nn_prop(p_ij, param, fit_out, fit_in, prop, ndir, selfloops):
        """ Compute the expected average nearest neighbour property.
        """
        av_nn = np.zeros(prop.shape, dtype=np.float64)
        for i in range(len(fit_out)):
            f_out_i = fit_out[i]
            f_in_i = fit_in[i]
            for j in range(i):
                f_out_j = fit_out[j]
                f_in_j = fit_in[j]
                pij = p_ij(param[0], f_out_i, f_in_j)
                pji = p_ij(param[0], f_out_j, f_in_i)
                if ndir == 'out':
                    av_nn[i] += pij*prop[j]
                    av_nn[j] += pji*prop[i]
                elif ndir == 'in':
                    av_nn[i] += pji*prop[j]
                    av_nn[j] += pij*prop[i]
                elif ndir == 'out-in':
                    p = 1 - (1 - pij)*(1 - pji)
                    av_nn[i] += p*prop[j]
                    av_nn[j] += p*prop[i]
                else:
                    raise ValueError('Direction of neighbourhood not right.')

        if selfloops:
            for i in range(len(fit_out)):
                pii = p_ij(param[0], fit_out[i], fit_in[i])
                if ndir == 'out':
                    av_nn[i] += pii*prop[i]
                elif ndir == 'in':
                    av_nn[i] += pii*prop[i]
                elif ndir == 'out-in':
                    av_nn[i] += pii*prop[i]
                else:
                    raise ValueError('Direction of neighbourhood not right.')

        return av_nn

    @staticmethod
    @jit(nopython=True)
    def _likelihood(p_ij, param, fit_out, fit_in, adj_i, adj_j, selfloops):
        """ Compute the binary log likelihood of a graph given the fitted model.
        """
        like = 0
        for i in range(len(fit_out)):
            f_out_i = fit_out[i]
            n = adj_i[i]
            m = adj_i[i+1]
            j_list = adj_j[n:m]
            for j in range(len(fit_in)):
                f_in_j = fit_in[j]
                if (i != j) | selfloops:
                    p = p_ij(param[0], f_out_i, f_in_j)
                    # Check if link exists
                    if j in j_list:
                        like += log(p)
                    else:
                        like += log(1 - p)
        
        return like

    @staticmethod
    @jit(nopython=True)
    def _binary_sample(p_ij, param, fit_out, fit_in, selfloops):
        """ Sample from the ensemble.
        """
        rows = []
        cols = []
        for i, f_out_i in enumerate(fit_out):
            for j, f_in_j in enumerate(fit_in):
                if (i != j) | selfloops:
                    p = p_ij(param[0], f_out_i, f_in_j)
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)

        return rows, cols

    @staticmethod
    @jit(nopython=True)
    def _cremb_sample(p_ij, param, fit_out, fit_in, s_out, s_in, selfloops):
        """ Sample from the ensemble with weights from the CremB model.
        """
        s_tot = np.sum(s_out)
        msg = 'Sum of in/out strengths not the same.'
        assert np.abs(1 - np.sum(s_in)/s_tot) < 1e-6, msg
        rows = []
        cols = []
        vals = []
        for i, f_out_i in enumerate(fit_out):
            for j, f_in_j in enumerate(fit_in):
                if (i != j) | selfloops:
                    p = p_ij(param[0], f_out_i, f_in_j)
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        vals.append(rng.exponential(
                            s_out[i]*s_in[j]/(s_tot*p)))

        return rows, cols, vals
