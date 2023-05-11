""" This module defines the classes that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

from . import graphs
from . import methods as mt
import numpy as np
import numpy.random as rng
import scipy.sparse as sp
import warnings
from numba import jit
from math import floor
from math import exp
from math import log
from math import isinf


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


class FitnessModel():
    """ The Fitness model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    sc: Spark Context
        the Spark Context
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
    p_blocks: int
        the number of blocks in which the fitnesses will be
        divided for parallel computation, note that the number
        of elements of the rdd will be p_blocks**2
    selfloops: bool
        selects if self loops (connections from i to i) are allowed
    """

    def __init__(self, sc, *args, **kwargs):
        """ Return a FitnessModel for the given graph data.
        The model accepts as arguments either: a WeightedGraph,
        in which case the strengths are used as fitnesses, or
        directly the fitness sequences (in and out).
        The model accepts the fitness sequences as numpy arrays.
        """
        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.WeightedGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges
                self.id_dtype = g.id_dtype
                self.fit_out = g.out_strength(get=True)
                self.fit_in = g.in_strength(get=True)
            else:
                raise ValueError('First argument passed must be a '
                                 'WeightedGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'fit_out',
                             'fit_in', 'param', 'p_blocks', 'selfloops']
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
            self.selfloops = True

        if not hasattr(self, 'id_dtype'):
            num_bytes = mt.get_num_bytes(self.num_vertices)
            self.id_dtype = np.dtype('u' + str(num_bytes))

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
        
        # Ensure that the number of blocks is a positive integer
        if not hasattr(self, 'p_blocks'):
            self.p_blocks = 10
        else:
            try: 
                assert self.p_blocks / int(self.p_blocks) == 1
                self.p_blocks = int(self.p_blocks)
            except Exception:
                raise ValueError(
                    'Number of parallel blocks must be an integer.')

            if self.p_blocks <= 0:
                raise ValueError(
                    'Number of parallel blocks must be a positive number.')

        # Create two RDDs to parallelize computations
        # The first simply divides the pij matrix in blocks
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i*step, self.num_vertices)
            else:
                x = (i*step, (i+1)*step)
            for j in range(self.p_blocks):
                if j == self.p_blocks - 1:
                    y = (j*step, self.num_vertices)
                else:
                    y = (j*step, (j+1)*step)
                elements.append(((x, y), self.fit_out[x[0]:x[1]], 
                                self.fit_in[y[0]:y[1]]))

        self.p_iter_rdd = sc.parallelize(
            elements, numSlices=len(elements)).cache()

        # the second has a triangular structure allowing
        # to iterate over pij and pji in the same block
        elements = []
        step = floor(self.num_vertices / self.p_blocks)
        for i in range(self.p_blocks):
            if i == self.p_blocks - 1:
                x = (i*step, self.num_vertices)
            else:
                x = (i*step, (i+1)*step)
            for j in range(i + 1):
                if j == self.p_blocks - 1:
                    y = (j*step, self.num_vertices)
                else:
                    y = (j*step, (j+1)*step)
                elements.append((x, y))

        self.p_sym_rdd = sc.parallelize(
            elements, numSlices=len(elements)).cache()

        # Assign to each parallel partition the correct fitness values
        fin = self.fit_in
        fout = self.fit_out
        fmap = self.fit_map
        self.p_sym_rdd = self.p_sym_rdd.map(lambda x: fmap(x, fout, fin))

    def fit(self, x0=None, method='density', solver='Newton-CG', atol=1e-18, 
            rtol=1e-9, maxiter=100, verbose=False):
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
        rtol : float
            relative tolerance for the exit condition
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
                'The ScaleInvariantModel requires one parameter.')

        if not np.issubdtype(x0.dtype, np.number):
            raise ValueError('x0 must be numeric.')

        if np.any(x0 < 0):
            raise ValueError('x0 must be positive.')

        if method == 'density':
            # Ensure that num_edges is set
            if not hasattr(self, 'num_edges'):
                raise ValueError(
                    'Number of edges must be set for density solver.')
            sol = mt.monotonic_newton_solver(
                x0, self.density_fit_fun, tol=atol, xtol=rtol, max_iter=100,
                full_return=True, verbose=verbose)

        elif method == 'mle':
            raise ValueError("Method not implemented.")

        else:
            raise ValueError("The selected method is not valid.")

        # Update results and check convergence
        self.param = sol.x
        self.solver_output = sol

        if not self.solver_output.converged:
            warnings.warn('Fit did not converge', UserWarning)

    def expected_num_edges(self, get=False):
        """ Compute the expected number of edges.
        """
        if not hasattr(self, 'param'):
            raise Exception('Model must be fitted beforehand.')
        
        # It is necessary to select the elements or pickling will fail
        e_fun = self.exp_edges
        p_ij = self.p_ij
        delta = self.param
        slflp = self.selfloops
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(p_ij, delta, x[0][0], x[0][1], x[1], x[2], slflp))
        self.exp_num_edges = tmp.fold(0, lambda x, y: x + y)

        if get:
            return self.exp_num_edges

    def expected_degrees(self, get=False):
        """ Compute the expected undirected/out/in degree.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted beforehand.')
        
        # It is necessary to select the elements or pickling will fail
        e_fun = self.exp_degrees
        p_ij = self.p_ij
        delta = self.param
        slflp = self.selfloops
        num_v = self.num_vertices
        tmp = self.p_sym_rdd.map(
            lambda x: e_fun(
                p_ij, delta, x[0][0], x[0][1], x[1], x[2], num_v, slflp))
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)
        res = tmp.fold((exp_d, exp_d_out, exp_d_in), 
                       lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
        self.exp_degree = res[0] 
        self.exp_out_degree = res[1]
        self.exp_in_degree = res[2]

        if get:
            return self.exp_degree, self.exp_out_degree, self.exp_in_degree
        
    def expected_degree(self, get=False):
        """ Compute the expected undirected degree for a given z.
        """
        self.expected_degrees()

        if get:
            return self.exp_degree

    def expected_out_degree(self, get=False):
        """ Compute the expected out degree for a given z.
        """
        self.expected_degrees()

        if get:
            return self.exp_out_degree

    def expected_in_degree(self, get=False):
        """ Compute the expected in degree for a given z.
        """
        self.expected_degrees()
        
        if get:
            return self.exp_in_degree

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
        if deg_recompute or not hasattr(self, 'exp_out_degree'):
            self.expected_degrees()

        if ndir == 'out':
            deg = self.exp_out_degree
        elif ndir == 'in':
            deg = self.exp_in_degree
        elif ndir == 'out-in':
            deg = self.exp_degree
        else:
            raise ValueError('Neighbourhood direction not recognised.')

        # It is necessary to select the elements or pickling will fail
        e_fun = self.exp_av_nn_prop
        p_ij = self.p_ij
        delta = self.param
        tmp = self.p_sym_rdd.map(
            lambda x: e_fun(p_ij, delta, x[0][0], x[0][1], 
                            x[1], x[2], prop, ndir, selfloops))
        av_nn = tmp.fold(np.zeros(prop.shape, dtype=np.float64), 
                         lambda x, y: x + y)
        
        # Test that mask is the same
        ind = deg != 0
        msg = 'Got a av_nn for an empty neighbourhood.'
        assert np.all(av_nn[~ind] == 0), msg
        
        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        return av_nn

    def expected_av_nn_degree(self, ddir='out', ndir='out', selfloops=False,
                              deg_recompute=False, get=False):
        """ Computes the expected value of the nearest neighbour average of
        the degree.
        """
        # Compute correct expected degree
        if deg_recompute or not hasattr(self, 'exp_out_degree'):
            self.expected_degrees()

        if ddir == 'out':
            deg = self.exp_out_degree
        elif ddir == 'in':
            deg = self.exp_in_degree
        elif ddir == 'out-in':
            deg = self.exp_degree
        else:
            raise ValueError('Neighbourhood direction not recognised.')

        # Compute property and set attribute
        name = ('exp_av_' + ndir.replace('-', '_') + 
                '_nn_d_' + ddir.replace('-', '_'))
        res = self.expected_av_nn_property(
            deg, ndir=ndir, selfloops=selfloops, deg_recompute=False)
        setattr(self, name, res)

        if get:
            return getattr(self, name)

    def log_likelihood(self, g, selfloops=None):
        """ Compute the likelihood a graph given the fitted model.
        Accepts as input either a graph or an adjacency matrix.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before.')

        if selfloops is None:
            selfloops = self.selfloops

        if isinstance(g, graphs.DirectedGraph):
            # Extract binary adjacency matrix from graph
            adj = g.adjacency_matrix(kind='csr')
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
        e_fun = self._likelihood
        p_ij = self.p_ij
        delta = self.param
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(p_ij, delta, x[0][0], x[0][1], 
                            x[1], x[2], adj.indptr, adj.indices,
                            selfloops))
        like = tmp.fold(0, lambda x, y: x + y)

        return like

    def sample(self, selfloops=None):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before sampling.')

        if selfloops is None:
            selfloops = self.selfloops

        # Generate uninitialised graph object
        g = graphs.DirectedGraph.__new__(graphs.DirectedGraph)

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        g.id_dtype = self.id_dtype
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges and extract properties
        e_fun = self._sample
        p_ij = self.p_ij
        delta = self.param
        app_fun = self.safe_append
        tmp = self.p_iter_rdd.map(
            lambda x: e_fun(p_ij, delta, x[0][0], x[0][1], 
                            x[1], x[2], selfloops))
        e = tmp.fold([], lambda x, y: app_fun(x, y))
        e = np.array(e,
                     dtype=[('src', 'f8'),
                            ('dst', 'f8')]).view(type=np.recarray)

        e = e.astype([('src', g.id_dtype),
                      ('dst', g.id_dtype)])
        g.sort_ind = np.argsort(e)
        g.e = e[g.sort_ind]
        g.num_edges = mt.compute_num_edges(g.e)

        return g

    def density_fit_fun(self, delta):
        """ Return the objective function value and the Jacobian
            for a given value of delta.
        """
        f_jac = self.exp_edges_f_jac
        p_jac_ij = self.p_jac_ij
        slflps = self.selfloops
        tmp = self.p_iter_rdd.map(
            lambda x: f_jac(
                p_jac_ij, delta, x[0][0], x[0][1], x[1], x[2], slflps))
        f, jac = tmp.fold((0, 0), lambda x, y: (x[0] + y[0], x[1] + y[1]))
        f -= self.num_edges
        return f, jac

    @staticmethod
    def safe_append(x, y):
        res = []
        res.extend(x)
        res.extend(y)
        return res

    @staticmethod
    def fit_map(ind, x, y):
        """ Assigns to each partition the correct values of strengths to allow
            computations in parallel over the pij matrix.

            Note that this is done to ensure that each partition can compute
            pij and pji in the same loop to be able to compute undirected 
            properties of the ensemble.

            Parameters
            ----------
            ind: tuple
                a tuple containing the slices of the index of fit_out (ind[0])
                and of fit_in (ind[1]) to iterate over
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
    def exp_edges_f_jac(p_jac_ij, param, ind_out, ind_in, fit_out, fit_in, 
                        selfloops):
        """ Compute the objective function of the density solver and its
        derivative.
        """
        f = 0.0
        jac = 0.0
        for i in range(ind_out[1]-ind_out[0]):
            f_out_i = fit_out[i]
            for j in range(ind_in[1]-ind_in[0]):
                f_in_j = fit_in[j]
                if (ind_out[0]+i != ind_in[0]+j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ij(param[0], f_out_i, f_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)
    def exp_edges(p_ij, param, ind_out, ind_in, fit_out, fit_in, selfloops):
        """ Compute the expected number of edges.
        """
        exp_e = 0.0
        for i in range(ind_out[1]-ind_out[0]):
            f_out_i = fit_out[i]
            for j in range(ind_in[1]-ind_in[0]):
                f_in_j = fit_in[j]
                if (ind_out[0]+i != ind_in[0]+j) | selfloops:
                    exp_e += p_ij(param[0], f_out_i, f_in_j)

        return exp_e

    @staticmethod
    @jit(nopython=True)
    def exp_degrees(p_ij, param, ind_out, ind_in, fit_out, fit_in, num_v, 
                    selfloops):
        """ Compute the expected undirected, in and out degree sequences.
        """
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)
        if ind_out == ind_in:
            for i in range(ind_out[1]-ind_out[0]):
                ind_i = ind_out[0]+i
                f_out_i = fit_out[0][i]
                f_in_i = fit_in[1][i]
                for j in range(i+1):
                    ind_j = ind_in[0]+j
                    f_out_j = fit_out[1][j]
                    f_in_j = fit_in[0][j]
                    if ind_i != ind_j:
                        pij = p_ij(param[0], f_out_i, f_in_j)
                        pji = p_ij(param[0], f_out_j, f_in_i)
                        p = pij + pji - pij*pji
                        exp_d[ind_i] += p
                        exp_d[ind_j] += p
                        exp_d_out[ind_i] += pij
                        exp_d_out[ind_j] += pji
                        exp_d_in[ind_j] += pij
                        exp_d_in[ind_i] += pji
                    elif selfloops:
                        pii = p_ij(param[0], f_out_i, f_in_j)
                        exp_d[ind_i] += pii
                        exp_d_out[ind_i] += pii
                        exp_d_in[ind_j] += pii

        else:
            for i in range(ind_out[1]-ind_out[0]):
                ind_i = ind_out[0]+i
                f_out_i = fit_out[0][i]
                f_in_i = fit_in[1][i]
                for j in range(ind_in[1]-ind_in[0]):
                    ind_j = ind_in[0]+j
                    f_out_j = fit_out[1][j]
                    f_in_j = fit_in[0][j]
                    if ind_i != ind_j:
                        pij = p_ij(param[0], f_out_i, f_in_j)
                        pji = p_ij(param[0], f_out_j, f_in_i)
                        p = pij + pji - pij*pji
                        exp_d[ind_i] += p
                        exp_d[ind_j] += p
                        exp_d_out[ind_i] += pij
                        exp_d_out[ind_j] += pji
                        exp_d_in[ind_j] += pij
                        exp_d_in[ind_i] += pji
                    elif selfloops:
                        pii = p_ij(param[0], f_out_i, f_in_j)
                        exp_d[ind_i] += pii
                        exp_d_out[ind_i] += pii
                        exp_d_in[ind_j] += pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)
    def exp_av_nn_prop(p_ij, param, ind_out, ind_in, fit_out, fit_in, prop, 
                       ndir, selfloops):
        """ Compute the expected average nearest neighbour property.
        """
        av_nn = np.zeros(prop.shape, dtype=np.float64)
        if ind_out == ind_in:
            for i in range(ind_out[1]-ind_out[0]):
                ind_i = ind_out[0]+i
                f_out_i = fit_out[0][i]
                f_in_i = fit_in[1][i]
                for j in range(i+1):
                    ind_j = ind_in[0]+j
                    f_out_j = fit_out[1][j]
                    f_in_j = fit_in[0][j]
                    if ind_i != ind_j:
                        pij = p_ij(param[0], f_out_i, f_in_j)
                        pji = p_ij(param[0], f_out_j, f_in_i)
                        if ndir == 'out':
                            av_nn[ind_i] += pij*prop[ind_j]
                            av_nn[ind_j] += pji*prop[ind_i]
                        elif ndir == 'in':
                            av_nn[ind_i] += pji*prop[ind_j]
                            av_nn[ind_j] += pij*prop[ind_i]
                        elif ndir == 'out-in':
                            p = pij + pji - pij*pji
                            av_nn[ind_i] += p*prop[ind_j]
                            av_nn[ind_j] += p*prop[ind_i]
                        else:
                            raise ValueError(
                                'Direction of neighbourhood not right.')
                    elif selfloops:
                        pii = p_ij(param[0], f_out_i, f_in_j)
                        if ndir == 'out':
                            av_nn[ind_i] += pii*prop[ind_i]
                        elif ndir == 'in':
                            av_nn[ind_i] += pii*prop[ind_i]
                        elif ndir == 'out-in':
                            av_nn[ind_i] += pii*prop[ind_i]
                        else:
                            raise ValueError(
                                'Direction of neighbourhood not right.')

        else:
            for i in range(ind_out[1]-ind_out[0]):
                ind_i = ind_out[0]+i
                f_out_i = fit_out[0][i]
                f_in_i = fit_in[1][i]
                for j in range(ind_in[1]-ind_in[0]):
                    ind_j = ind_in[0]+j
                    f_out_j = fit_out[1][j]
                    f_in_j = fit_in[0][j]
                    if ind_i != ind_j:
                        pij = p_ij(param[0], f_out_i, f_in_j)
                        pji = p_ij(param[0], f_out_j, f_in_i)
                        if ndir == 'out':
                            av_nn[ind_i] += pij*prop[ind_j]
                            av_nn[ind_j] += pji*prop[ind_i]
                        elif ndir == 'in':
                            av_nn[ind_i] += pji*prop[ind_j]
                            av_nn[ind_j] += pij*prop[ind_i]
                        elif ndir == 'out-in':
                            p = pij + pji - pij*pji
                            av_nn[ind_i] += p*prop[ind_j]
                            av_nn[ind_j] += p*prop[ind_i]
                        else:
                            raise ValueError(
                                'Direction of neighbourhood not right.')
                    elif selfloops:
                        pii = p_ij(param[0], f_out_i, f_in_j)
                        if ndir == 'out':
                            av_nn[ind_i] += pii*prop[ind_i]
                        elif ndir == 'in':
                            av_nn[ind_i] += pii*prop[ind_i]
                        elif ndir == 'out-in':
                            av_nn[ind_i] += pii*prop[ind_i]
                        else:
                            raise ValueError(
                                'Direction of neighbourhood not right.')

        return av_nn

    @staticmethod
    @jit(nopython=True)
    def _likelihood(p_ij, param, ind_out, ind_in, fit_out, 
                    fit_in, adj_i, adj_j, selfloops):
        """ Compute the binary log likelihood of a graph given the fitted model.
        """
        like = 0
        for i in range(ind_out[1]-ind_out[0]):
            ind_i = ind_out[0]+i
            f_out_i = fit_out[i]
            n = adj_i[i]
            m = adj_i[i+1]
            j_list = adj_j[n:m]
            for j in range(ind_in[1]-ind_in[0]):
                ind_j = ind_in[0]+j
                f_in_j = fit_in[j]
                if (ind_i != ind_j) | selfloops:
                    p = p_ij(param[0], f_out_i, f_in_j)
                    # Check if link exists
                    if ind_j in j_list:
                        like += log(p)
                    else:
                        like += log(1 - p)
        
        return like

    @staticmethod
    @jit(nopython=True)
    def _sample(p_ij, param, ind_out, ind_in, fit_out, fit_in, selfloops):
        """ Sample from the ensemble.
        """
        sample = []
        for i in range(ind_out[1]-ind_out[0]):
            ind_i = ind_out[0]+i
            f_out_i = fit_out[i]
            for j in range(ind_in[1]-ind_in[0]):
                ind_j = ind_in[0]+j
                f_in_j = fit_in[j]
                if (ind_i != ind_j) | selfloops:
                    p = p_ij(param[0], f_out_i, f_in_j)
                    if rng.random() < p:
                        sample.append((ind_i, ind_j))

        return sample


class ScaleInvariantModel(FitnessModel):
    """ The Scale Invariant model takes the fitnesses of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    sc: Spark Context
        the Spark Context
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
    p_blocks: int
        the number of blocks in which the fitnesses will be
        divided for parallel computation, note that the number
        of elements of the rdd will be p_blocks**2
    selfloops: bool
        selects if self loops (connections from i to i) are allowed
    """

    def __init__(self, sc, *args, **kwargs):
        """ Return a ScaleInvariantModel for the given graph data.
        The model accepts as arguments either: a WeightedGraph,
        in which case the strengths are used as fitnesses, or
        directly the fitness sequences (in and out).
        The model accepts the fitness sequences as numpy arrays.
        """
        super().__init__(sc, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def p_ij(d, x_i, y_j):
        """ Compute the probability of connection between node i and j.
        """
        tmp = d*x_i*y_j
        if isinf(tmp):
            return 1.0
        else:
            return 1 - exp(-tmp)

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
            return 1 - exp(-tmp1), tmp * exp(-tmp1)

    @staticmethod
    @jit(nopython=True)
    def _likelihood(p_ij, param, ind_out, ind_in, fit_out, 
                    fit_in, adj_i, adj_j, selfloops):
        """ Compute the binary log likelihood of a graph given the fitted model.
        """
        like = 0
        for i in range(ind_out[1]-ind_out[0]):
            ind_i = ind_out[0]+i
            f_out_i = fit_out[i]
            n = adj_i[i]
            m = adj_i[i+1]
            j_list = adj_j[n:m]
            for j in range(ind_in[1]-ind_in[0]):
                ind_j = ind_in[0]+j
                f_in_j = fit_in[j]
                if (ind_i != ind_j) | selfloops:
                    # Check if link exists
                    if ind_j in j_list:
                        like += log(p_ij(param[0], f_out_i, f_in_j))
                    else:
                        like += param[0]*f_out_i*f_in_j
        
        return like
