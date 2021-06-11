""" This module defines the classes that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """

from . import graphs
from . import methods as mt
from . import lib
import numpy as np
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


class RandomGraph(GraphEnsemble):
    """ The simplest graph ensemble defined by conserving the total number of
    edges (per label) only. We assume the graph is directed.

    If it is initialized with a LabelGraph or with a number of labels, then
    the edges will be labelled.

    Attributes
    ----------
    num_vertices: int
        the total number of vertices
    num_labels: int or None
        the total number of labels by which the vector strengths are computed
    num_edges: float (or np.ndarray)
        the total number of edges (per label)
    total_weight: float (or np.ndarray)
        the sum of all edges weights (per label)
    p: float or np.ndarray
        the probability of each link (by label)
    q: float or np.ndarray
        the parameter defining the probability distribution of weights
    discrete_weights: boolean
        the flag determining if the distribution of weights is discrete or
        continuous
    """

    def __init__(self, *args, **kwargs):
        """ Return a RandomGraph ensemble.
        """

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.Graph):
                g = args[0]
                self.num_vertices = g.num_vertices
                if isinstance(g, graphs.LabelGraph):
                    self.num_edges = g.num_edges_label
                    self.num_labels = g.num_labels
                else:
                    self.num_edges = g.num_edges
                    self.num_labels = None
            else:
                ValueError('First argument passed must be a Graph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'num_labels',
                             'total_weight', 'p', 'q', 'discrete_weights']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)
            else:
                setattr(self, name, kwargs[name])

        # Check that all necessary attributes have been passed
        if not hasattr(self, 'num_vertices'):
            raise ValueError('Number of vertices not set.')

        if hasattr(self, 'p'):
            if hasattr(self, 'num_edges'):
                raise ValueError('Either p or num_edges can be set not both.')
            else:
                if not hasattr(self, 'num_labels'):
                    if isinstance(self.p, np.ndarray):
                        if len(self.p) > 1:
                            self.num_labels = len(self.p)
                        else:
                            self.num_labels = None
                    else:
                        self.num_labels = None

            msg = ('Number of p parameters must be the same as number'
                   ' of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.p), msg
            else:
                assert isinstance(self.p, (int, float)), msg
            self.num_edges = self.exp_num_edges()

        else:
            if not hasattr(self, 'num_edges'):
                raise ValueError('Neither p nor num_edges have been set.')

            if not hasattr(self, 'num_labels'):
                if isinstance(self.num_edges, np.ndarray):
                    if len(self.num_edges) > 1:
                        self.num_labels = len(self.num_edges)
                    else:
                        self.num_labels = None
                        self.num_edges = self.num_edges[0]
                else:
                    self.num_labels = None

            msg = ('Number of edges must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.num_edges), msg
            else:
                try:
                    int(self.num_edges)
                except Exception:
                    assert False, msg

        # Check if weight information is present
        if not hasattr(self, 'discrete_weights') and (
         hasattr(self, 'q') or hasattr(self, 'total_weight')):
            self.discrete_weights = False

        if hasattr(self, 'total_weight'):
            if hasattr(self, 'q'):
                msg = 'Either total_weight or q can be set not both.'
                raise Exception(msg)
            else:
                msg = ('total_weight must be a vector with length equal to '
                       'the number of labels.')
                if self.num_labels is not None:
                    assert self.num_labels == len(self.total_weight), msg
                else:
                    try:
                        int(self.num_edges)
                    except Exception:
                        assert False, msg

        elif hasattr(self, 'q'):
            msg = ('q must be a vector with length equal to '
                   'the number of labels.')
            if self.num_labels is not None:
                assert self.num_labels == len(self.q), msg
            else:
                try:
                    int(self.q)
                except Exception:
                    assert False, msg

            self.total_weight = self.exp_total_weight()

    def fit(self):
        """ Fit the parameter p and q to the number of edges and total weight.
        """
        self.p = self.num_edges/(self.num_vertices*(self.num_vertices - 1))

        if hasattr(self, 'total_weight'):
            if self.discrete_weights:
                self.q = 1 - self.num_edges/self.total_weight
            else:
                self.q = self.num_edges/self.total_weight

    def exp_num_edges(self):
        """ Compute the expected number of edges (per label) given p.
        """
        return self.p*self.num_vertices*(self.num_vertices - 1)

    def exp_total_weight(self):
        """ Compute the expected total weight (per label) given q.
        """
        if self.discrete_weights:
            return self.num_edges/(1 - self.q)
        else:
            return self.num_edges/self.q

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'p'):
            raise Exception('Ensemble has to be fitted before sampling.')

        # Generate uninitialised graph object
        if self.num_labels is None:
            if hasattr(self, 'q'):
                g = graphs.WeightedGraph.__new__(graphs.WeightedGraph)
            else:
                g = graphs.DirectedGraph.__new__(graphs.DirectedGraph)
        else:
            if hasattr(self, 'q'):
                g = graphs.WeightedLabelGraph.__new__(
                    graphs.WeightedLabelGraph)
            else:
                g = graphs.LabelGraph.__new__(graphs.LabelGraph)
            g.lv = graphs.LabelVertexList()

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = mt.get_num_bytes(self.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges
        if self.num_labels is None:
            if hasattr(self, 'q'):
                e = mt.random_graph(self.num_vertices, self.p,
                                    self.q, self.discrete_weights)
                e = e.view(type=np.recarray,
                           dtype=[('src', 'f8'),
                                  ('dst', 'f8'),
                                  ('weight', 'f8')]).reshape((e.shape[0],))
                e = e.astype([('src', g.id_dtype),
                              ('dst', g.id_dtype),
                              ('weight', 'f8')])
                g.total_weight = np.sum(e.weight)

            else:
                e = mt.random_graph(self.num_vertices, self.p)
                e = e.view(type=np.recarray,
                           dtype=[('src', 'f8'),
                                  ('dst', 'f8')]).reshape((e.shape[0],))
                e = e.astype([('src', g.id_dtype), ('dst', g.id_dtype)])

            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = mt.compute_num_edges(g.e)

        else:
            if hasattr(self, 'q'):
                e = mt.random_labelgraph(self.num_vertices,
                                         self.num_labels,
                                         self.p,
                                         self.q,
                                         self.discrete_weights)
                e = e.view(type=np.recarray,
                           dtype=[('label', 'f8'),
                                  ('src', 'f8'),
                                  ('dst', 'f8'),
                                  ('weight', 'f8')]).reshape((e.shape[0],))
                g.num_labels = self.num_labels
                num_bytes = mt.get_num_bytes(g.num_labels)
                g.label_dtype = np.dtype('u' + str(num_bytes))

                e = e.astype([('label', g.label_dtype),
                              ('src', g.id_dtype),
                              ('dst', g.id_dtype),
                              ('weight', 'f8')])
                g.total_weight = np.sum(e.weight)
                g.total_weight_label = mt.compute_tot_weight_by_label(
                    e, self.num_labels)

            else:
                e = mt.random_labelgraph(self.num_vertices,
                                         self.num_labels,
                                         self.p)
                e = e.view(type=np.recarray,
                           dtype=[('label', 'f8'),
                                  ('src', 'f8'),
                                  ('dst', 'f8')]).reshape((e.shape[0],))
                g.num_labels = self.num_labels
                num_bytes = mt.get_num_bytes(g.num_labels)
                g.label_dtype = np.dtype('u' + str(num_bytes))

                e = e.astype([('label', g.label_dtype),
                              ('src', g.id_dtype),
                              ('dst', g.id_dtype)])

            g.sort_ind = np.argsort(e)
            g.e = e[g.sort_ind]
            g.num_edges = mt.compute_num_edges(g.e)
            ne_label = mt.compute_num_edges_by_label(g.e, g.num_labels)
            dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
            g.num_edges_label = ne_label.astype(dtype)

        return g


class FitnessModel(GraphEnsemble):
    """ The fitness model takes the strengths of each node in order to
    construct a probability distribution over all possible graphs.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength sequence
    in_strength: np.ndarray
        the in strength sequence
    num_edges: int
        the total number of edges
    num_vertices: int
        the total number of nodes
    z: float
        the density parameter
    """

    def __init__(self, *args, scale_invariant=False, min_degree=False,
                 **kwargs):
        """ Return a FitnessModel for the given graph data.

        The model accepts as arguments either: a WeightedGraph, the
        strength sequences (in and out) and the number of edges,
        or the strength sequences and the z parameter.

        The model accepts the strength sequences as numpy recarrays. The first
        column must contain the node index to which the strength refers and
        in the second column there must be the value of the strength.
        All missing node ids are assumed zero.

        The model defaults to the classic fitness model functional but can be
        set to use the scale invariant formulation by setting the
        scale_invariant flag to True. This changes the functional used for
        the computation of the link probability but nothing else.
        """
        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.WeightedGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges
                self.id_dtype = g.id_dtype
                self.out_strength = g.out_strength(get=True)
                self.in_strength = g.in_strength(get=True)
            else:
                raise ValueError('First argument passed must be a '
                                 'WeightedGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        self.scale_invariant = scale_invariant
        self.min_degree = min_degree

        allowed_arguments = ['num_vertices', 'num_edges', 'out_strength',
                             'in_strength', 'param', 'discrete_weights']
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

        if not hasattr(self, 'out_strength'):
            raise ValueError('out_strength not set.')

        if not hasattr(self, 'in_strength'):
            raise ValueError('in_strength not set.')

        if not (hasattr(self, 'num_edges') or
                hasattr(self, 'param')):
            raise ValueError('Either num_edges or param must be set.')

        if not hasattr(self, 'id_dtype'):
            num_bytes = mt.get_num_bytes(self.num_vertices)
            self.id_dtype = np.dtype('u' + str(num_bytes))

        # Ensure that strengths passed adhere to format (ndarray)
        msg = ("Out strength must be a numpy array of length " +
               str(self.num_vertices))
        assert isinstance(self.out_strength, np.ndarray), msg
        assert self.out_strength.shape == (self.num_vertices,), msg

        msg = ("In strength must be a numpy array of length " +
               str(self.num_vertices))
        assert isinstance(self.in_strength, np.ndarray), msg
        assert self.in_strength.shape == (self.num_vertices,), msg

        # Ensure that strengths have positive values only
        msg = "Out strength must contain positive values only."
        assert np.all(self.out_strength >= 0), msg

        msg = "In strength must contain positive values only."
        assert np.all(self.in_strength >= 0), msg

        # Ensure that number of edges is a positive number
        if hasattr(self, 'num_edges'):
            if not isinstance(self.num_edges, (int, float)):
                raise ValueError('Number of edges must be a number.')

            if self.num_edges < 0:
                raise ValueError(
                    'Number of edges must be a positive number.')
        else:
            if not isinstance(self.param, np.ndarray):
                self.param = np.array([self.param])

            if self.min_degree:
                if not (len(self.param) == 2):
                    raise ValueError('The FitnessModel with min degree '
                                     'correction requires two parameters.')
            else:
                if not (len(self.param) == 1):
                    raise ValueError(
                        'The FitnessModel requires one parameter.')
            
            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError('Parameters must be numeric.')

            if np.any(self.param < 0):
                raise ValueError('Parameters must be positive.')

        # Check that sum of in and out strengths are equal
        msg = 'Sums of strengths do not match.'
        assert np.allclose(np.sum(self.out_strength),
                           np.sum(self.in_strength),
                           atol=1e-14, rtol=1e-9), msg

        # Get the correct probability functional
        if scale_invariant:
            if min_degree:
                msg = 'Cannot constrain min degree in scale invariant model.'
                raise ValueError(msg)
            else:
                self.prob_fun = mt.p_invariant
                self.jac_fun = mt.jac_invariant
        else:
            if min_degree:
                self.prob_fun = mt.p_fitness_alpha
                self.jac_fun = mt.jac_fitness_alpha
            else:
                self.prob_fun = mt.p_fitness
                self.jac_fun = mt.jac_fitness

        # If param are set computed expected number of edges per label
        if hasattr(self, 'param'):
            self.num_edges = mt.fit_exp_edges(
                self.prob_fun,
                self.param,
                self.out_strength,
                self.in_strength)

    def fit(self, x0=None, method=None, tol=1e-5, xtol=1e-12, max_iter=100,
            verbose=False):
        """ Compute the optimal z to match the given number of edges.

        Parameters
        ----------
        x0: float
            optional initial conditions for parameters
        method: 'newton' or 'fixed-point'
            selects which method to use for the solver
        tol : float
            tolerance for the exit condition on the norm
        eps : float
            tolerance for the exit condition on difference between two
            iterations
        max_iter : int or float
            maximum number of iteration
        verbose: boolean
            if true print debug info while iterating

        """
        if method is None:
            if not self.min_degree:
                method = 'newton'
        elif self.min_degree:
            warnings.warn('Method not recognised for solver with min degree '
                          'constraint, using default SLSQP.', UserWarning)

        if (method == 'fixed-point') and self.scale_invariant:
            raise Exception('Fixed point solver not supported for scale '
                            'invariant functional.')

        if x0 is None:
            if self.min_degree:
                x0 = np.array([0, 1], dtype=np.float64)
            else:
                x0 = np.array([0], dtype=np.float64)
        else:
            if not isinstance(x0, np.ndarray):
                try:
                    x0 = np.array([x for x in x0])
                except Exception:
                    x0 = np.array([x0])

            if self.min_degree:
                if not (len(x0) == 2):
                    raise ValueError('The FitnessModel with min degree '
                                     'correction requires two parameters.')
            else:
                if not (len(x0) == 1):
                    raise ValueError(
                        'The FitnessModel requires one parameter.')

            if not np.issubdtype(x0.dtype, np.number):
                raise ValueError('x0 must be numeric.')

            if np.any(x0 < 0):
                raise ValueError('x0 must be positive.')

        if self.min_degree:
            # Find min degree node
            min_out_i = np.argmin(self.out_strength[self.out_strength > 0])
            min_in_i = np.argmin(self.in_strength[self.in_strength > 0])

            def min_d(x):
                return np.array([mt.fit_ineq_constr_alpha(
                                    x, self.prob_fun, min_out_i,
                                    self.out_strength[min_out_i],
                                    self.in_strength),
                                 mt.fit_ineq_constr_alpha(
                                    x, self.prob_fun, min_in_i,
                                    self.in_strength[min_in_i],
                                    self.out_strength)
                                 ], dtype=np.float64)

            def jac_min_d(x):
                return np.stack(
                    (mt.fit_ineq_jac_alpha(
                        x, self.jac_fun, min_out_i,
                        self.out_strength[min_out_i],
                        self.in_strength),
                     mt.fit_ineq_jac_alpha(
                        x, self.jac_fun, min_in_i,
                        self.in_strength[min_in_i],
                        self.out_strength)),
                    axis=0)

            # Solve
            sol = mt.alpha_solver(
                x0=x0,
                fun=lambda x: mt.fit_eq_constr_alpha(
                    x, self.prob_fun, self.out_strength,
                    self.in_strength, self.num_edges),
                jac=lambda x: mt.fit_eq_jac_alpha(
                    x, self.jac_fun, self.out_strength,
                    self.in_strength),
                min_d=min_d,
                jac_min_d=jac_min_d,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                full_return=True)

        elif method == "newton":
            sol = mt.newton_solver(
                x0=x0,
                fun=lambda x: mt.fit_f_jac(
                    self.prob_fun, self.jac_fun, x,
                    self.out_strength, self.in_strength, self.num_edges),
                tol=tol,
                xtol=xtol,
                max_iter=max_iter,
                verbose=verbose,
                full_return=True)

        elif method == "fixed-point":
            sol = mt.fixed_point_solver(
                x0=x0,
                fun=lambda x: mt.fit_iterative(
                    x, self.out_strength, self.in_strength, self.num_edges),
                xtol=xtol,
                max_iter=max_iter,
                verbose=verbose,
                full_return=True)

        else:
            raise ValueError("The selected method is not valid.")

        # Update results and check convergence
        self.param = sol.x
        self.solver_output = sol

        if not sol.converged:
            warnings.warn('Fit did not converge', UserWarning)

    def expected_num_edges(self):
        """ Compute the expected number of edges (per label).
        """
        if not hasattr(self, 'exp_num_edges'):
            if not hasattr(self, 'param'):
                raise Exception('Model must be fitted before hand.')
            
            self.exp_num_edges = mt.fit_exp_edges(
                    self.prob_fun,
                    self.param,
                    self.out_strength,
                    self.in_strength)

        return self.exp_num_edges

    def expected_out_degree(self):
        """ Compute the expected out degree for a given z.
        """
        if not hasattr(self, 'exp_out_degree'):
            if not hasattr(self, 'param'):
                raise Exception('Ensemble has to be fitted before sampling.')

            self.exp_out_degree, self.exp_in_degree = mt.fit_exp_degree(
                self.prob_fun, self.param, self.out_strength,
                self.in_strength)

        return self.exp_out_degree

    def expected_in_degree(self):
        """ Compute the expected in degree for a given z.
        """
        if not hasattr(self, 'exp_in_degree'):
            _ = self.expected_out_degree()

        return self.exp_in_degree

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before sampling.')

        # Generate uninitialised graph object
        g = graphs.WeightedGraph.__new__(graphs.WeightedGraph)

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        g.id_dtype = self.id_dtype
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges and extract properties
        e = mt.fit_sample(
                self.prob_fun, self.param, self.out_strength, self.in_strength)

        e = np.array(e,
                     dtype=[('src', 'f8'),
                            ('dst', 'f8'),
                            ('weight', 'f8')]).view(type=np.recarray)

        e = e.astype([('src', g.id_dtype),
                      ('dst', g.id_dtype),
                      ('weight', 'f8')])
        g.sort_ind = np.argsort(e)
        g.e = e[g.sort_ind]
        g.num_edges = mt.compute_num_edges(g.e)
        g.total_weight = np.sum(e.weight)

        return g


class StripeFitnessModel(GraphEnsemble):
    """ A generalized fitness model that allows for strengths by label.

    This model allows to take into account labels of the edges and include
    this information as part of the model. The strength sequence is therefore
    now subdivided in strength per label. Two quantities can be preserved by
    the ensemble: either the total number of edges, or the number of edges per
    label.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength sequence
    in_strength: np.ndarray
        the in strength sequence
    num_edges: int (or np.ndarray)
        the total number of edges (per label)
    num_vertices: int
        the total number of nodes
    num_labels: int
        the total number of labels by which the vector strengths are computed
    param: np.ndarray
        the parameters vector
    """

    def __init__(self, *args, per_label=True, scale_invariant=False,
                 min_degree=False, **kwargs):
        """ Return a StripeFitnessModel for the given graph data.

        The model accepts as arguments either: a WeightedLabelGraph, the
        strength sequences (in and out) and the number of edges (per label),
        or the strength sequences and the z parameter (per label).

        The model accepts the strength sequences as numpy recarrays. The first
        column must contain the label index, the second column the node index
        to which the strength refers, and in the third column must have the
        value of the strength for the node label pair. All node label pairs
        not included are assumed zero.

        The model defaults to the classic fitness model functional but can be
        set to use the scale invariant formulation by setting the
        scale_invariant flag to True. This changes the functional used for
        the computation of the link probability but nothing else.

        Note that the number of edges given implicitly determines if the
        quantity preserved is the total number of edges or the number of edges
        per label. Pass only one integer for the first and a numpy array for
        the second. Note that if an array is passed then the index must be the
        same as the one in the strength sequence.

        """

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.WeightedLabelGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_labels = g.num_labels
                self.id_dtype = g.id_dtype
                self.label_dtype = g.label_dtype
                self.out_strength = g.out_strength_by_label(get=True)
                self.in_strength = g.in_strength_by_label(get=True)
                if per_label:
                    self.num_edges = g.num_edges_label
                else:
                    self.num_edges = g.num_edges
            else:
                raise ValueError('First argument passed must be a '
                                 'WeightedLabelGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        self.scale_invariant = scale_invariant
        self.min_degree = min_degree

        allowed_arguments = ['num_vertices', 'num_edges', 'num_labels',
                             'out_strength', 'in_strength', 'param',
                             'discrete_weights']
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

        if not hasattr(self, 'num_labels'):
            raise ValueError('Number of labels not set.')
        else:
            try: 
                assert self.num_labels / int(self.num_labels) == 1
                self.num_labels = int(self.num_labels)
            except Exception:
                raise ValueError('Number of labels must be an integer.')

            if self.num_labels <= 0:
                raise ValueError(
                    'Number of labels must be a positive number.')

        if not hasattr(self, 'out_strength'):
            raise ValueError('out_strength not set.')

        if not hasattr(self, 'in_strength'):
            raise ValueError('in_strength not set.')

        if not (hasattr(self, 'num_edges') or
                hasattr(self, 'param')):
            raise ValueError('Either num_edges or param must be set.')

        if not hasattr(self, 'id_dtype'):
            num_bytes = mt.get_num_bytes(self.num_vertices)
            self.id_dtype = np.dtype('u' + str(num_bytes))

        if not hasattr(self, 'label_dtype'):
            num_bytes = mt.get_num_bytes(self.num_labels)
            self.label_dtype = np.dtype('u' + str(num_bytes))

        # Ensure that strengths passed adhere to format
        msg = ("Out strength must be a rec array with columns: "
               "('label', 'id', 'value')")
        assert isinstance(self.out_strength, np.recarray), msg
        for clm in self.out_strength.dtype.names:
            assert clm in ('label', 'id', 'value'), msg

        msg = ("In strength must be a rec array with columns: "
               "('label', 'id', 'value')")
        assert isinstance(self.in_strength, np.recarray), msg
        for clm in self.in_strength.dtype.names:
            assert clm in ('label', 'id', 'value'), msg

        # Ensure that num_vertices and num_labels are coherent with info
        # in strengths
        if self.num_vertices < max(np.max(self.out_strength.id),
                                   np.max(self.in_strength.id)):
            raise ValueError(
                'Number of vertices smaller than max id value in strengths.')

        if self.num_labels < max(np.max(self.out_strength.label),
                                 np.max(self.in_strength.label)):
            raise ValueError(
                'Number of labels smaller than max label value in strengths.')

        # Ensure that all labels, ids and values of the strength are positive
        if np.any(self.out_strength.label < 0):
            raise ValueError(
                "Out strength labels must contain positive values only.")

        if np.any(self.in_strength.label < 0):
            raise ValueError(
                "In strength labels must contain positive values only.")

        if np.any(self.out_strength.id < 0):
            raise ValueError(
                "Out strength ids must contain positive values only.")

        if np.any(self.in_strength.id < 0):
            raise ValueError(
                "In strength ids must contain positive values only.")

        if np.any(self.out_strength.value < 0):
            raise ValueError(
                "Out strength values must contain positive values only.")

        if np.any(self.in_strength.value < 0):
            raise ValueError(
                "In strength values must contain positive values only.")

        msg = "Storing zeros in the strengths leads to inefficient code."
        if np.any(self.out_strength.value == 0) or np.any(
                self.in_strength.value == 0):
            msg = "Storing zeros in the strengths leads to inefficient code."
            warnings.warn(msg, UserWarning)

        # Ensure that strengths are sorted
        self.out_strength = self.out_strength[['label', 'id', 'value']]
        self.in_strength = self.in_strength[['label', 'id', 'value']]
        self.out_strength.sort()
        self.in_strength.sort()

        # Ensure that number of constraint matches number of labels
        if hasattr(self, 'num_edges'):
            if not isinstance(self.num_edges, np.ndarray):
                self.num_edges = np.array([self.num_edges])

            msg = ('Number of edges must be a number or a numpy array of '
                   'length equal to the number of labels.')
            if len(self.num_edges) > 1:
                self.per_label = True
                assert len(self.num_edges) == self.num_labels, msg
            else:
                self.per_label = False

            if not np.issubdtype(self.num_edges.dtype, np.number):
                raise ValueError(msg)

            if np.any(self.num_edges < 0):
                msg = 'Number of edges must contain only positive values.'
                raise ValueError(msg)
        else:
            if not isinstance(self.param, np.ndarray):
                try:
                    self.param = np.array([p for p in self.param])
                except Exception:
                    self.param = np.array([self.param])

            # Ensure that param has two dimensions (row 1: z, row 2: alpha,
            # each column is a layer, if single z then single column)
            if self.param.ndim < 2:
                self.param = np.array([self.param])
            elif self.param.ndim > 2:
                raise ValueError('StripeFitnessModel parameters must have '
                                 'two dimensions max.')

            p_shape = self.param.shape
            if self.min_degree:
                msg = ('StripeFitnessModel with min degree correction requires'
                       ' two element or two rows with number of columns '
                       'equal to the number of labels.')
                if p_shape[0] != 2:
                    raise ValueError(msg)
                elif p_shape[1] == self.num_labels:
                    self.per_label = True
                elif p_shape[1] == 1:
                    self.per_label = False
                else:
                    raise ValueError(msg)
            else:
                msg = ('StripeFitnessModel requires an array of parameters '
                       'with number of elements equal to the number of labels '
                       'or to one.')
                if p_shape[0] != 1:
                    raise ValueError(msg)
                elif p_shape[1] == self.num_labels:
                    self.per_label = True
                elif p_shape[1] == 1:
                    self.per_label = False
                else:
                    raise ValueError(msg)

            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError('Parameters must be numeric.')

            if np.any(self.param < 0):
                raise ValueError('Parameters must be positive.')

        # Check that sum of in and out strengths are equal per label
        tot_out = np.zeros((self.num_labels))
        for row in self.out_strength:
            tot_out[row.label] += row.value
        tot_in = np.zeros((self.num_labels))
        for row in self.in_strength:
            tot_in[row.label] += row.value

        msg = 'Sums of strengths per label do not match.'
        assert np.allclose(tot_out, tot_in, atol=1e-14, rtol=1e-9), msg

        # Get the correct probability functional
        if scale_invariant:
            if min_degree:
                msg = 'Cannot constrain min degree in scale invariant model.'
                raise ValueError(msg)
            else:
                self.prob_fun = mt.p_invariant
                self.jac_fun = mt.jac_invariant
        else:
            if min_degree:
                self.prob_fun = mt.p_fitness_alpha
                self.jac_fun = mt.jac_fitness_alpha
            else:
                self.prob_fun = mt.p_fitness
                self.jac_fun = mt.jac_fitness

        # If param are set computed expected number of edges per label
        if hasattr(self, 'param'):
            if self.per_label:
                self.num_edges = self.expected_num_edges_label()
            else:
                self.num_edges = self.expected_num_edges()

    def fit(self, x0=None, method=None, tol=1e-5, xtol=1e-12, max_iter=100,
            verbose=False):
        """ Compute the optimal z to match the given number of edges.

        Parameters
        ----------
        x0: np.ndarray
            optional initial conditions for parameters
        method: 'newton' or 'fixed-point'
            selects which method to use for the solver
        tol : float
            tolerance for the exit condition on the norm
        eps : float
            tolerance for the exit condition on difference between two
            iterations
        max_iter : int or float
            maximum number of iteration
        verbose: boolean
            if true print debug info while iterating

        """
        if method is None:
            if not self.min_degree:
                method = 'newton'
        elif self.min_degree:
            warnings.warn('Method not recognised for solver with min degree '
                          'constraint, using default SLSQP.', UserWarning)

        if (method == 'fixed-point') and (self.scale_invariant or not
                                          self.per_label):
            raise Exception('Fixed point solver not supported for scale '
                            'invariant functional or for fit not per label.')

        # Ensure initial conditions x0 are of correct format
        if x0 is None:
            if self.per_label:
                num_clm = self.num_labels
            else:
                num_clm = 1

            if self.min_degree:
                x0 = np.zeros((2, num_clm), dtype=np.float64)
                x0[1, :] = 1
            else:
                x0 = np.zeros((1, num_clm), dtype=np.float64)
        else:
            if not isinstance(x0, np.ndarray):
                try:
                    x0 = np.array([x for x in x0])
                except Exception:
                    x0 = np.array([x0])

        # Ensure that x0 has two dimensions (row 1: z, row 2: alpha,
        # each column is a layer, if single z then single column)
        if x0.ndim < 2:
            x0 = np.array([x0])
        elif x0.ndim > 2:
            raise ValueError('StripeFitnessModel x0 must have '
                             'two dimensions max.')
        p_shape = x0.shape
        if self.min_degree:
            msg = ('StripeFitnessModel with min degree correction requires'
                   ' two element or two rows with number of columns '
                   'equal to the number of labels.')
            if p_shape[0] != 2:
                raise ValueError(msg)
            elif self.per_label:
                assert p_shape[1] == self.num_labels, msg
            else:
                assert p_shape[1] == 1, msg
        else:
            msg = ('StripeFitnessModel requires an array of parameters '
                   'with number of elements equal to the number of labels '
                   'or to one.')
            if p_shape[0] != 1:
                raise ValueError(msg)
            elif self.per_label:
                assert p_shape[1] == self.num_labels, msg
            else:
                assert p_shape[1] == 1, msg

        if not np.issubdtype(x0.dtype, np.number):
            raise ValueError('Parameters must be numeric.')

        if np.any(x0 < 0):
            raise ValueError('Parameters must be positive.')

        # Fit by layer
        if self.per_label:
            if self.min_degree:
                self.param = np.empty((2, self.num_labels), dtype=np.float64)
            else:
                self.param = np.empty((1, self.num_labels), dtype=np.float64)
            
            self.solver_output = [None]*self.num_labels

            for i in range(self.num_labels):
                s_out = self.out_strength[self.out_strength.label == i]
                s_in = self.in_strength[self.in_strength.label == i]
                num_e = self.num_edges[i]
                
                if self.min_degree:
                    # Find min degree node
                    min_out_i = np.argmin(s_out.value[s_out.value > 0])
                    min_in_i = np.argmin(s_in.value[s_in.value > 0])
                    min_out_id = s_out[min_out_i].id
                    min_in_id = s_in[min_in_i].id

                    def min_d(x):
                        return np.array([
                            mt.layer_ineq_constr_alpha(
                                x, self.prob_fun, min_out_id,
                                s_out.value[min_out_i], s_in),
                            mt.layer_ineq_constr_alpha(
                                x, self.prob_fun, min_in_id,
                                s_in.value[min_in_i], s_out)],
                            dtype=np.float64)

                    def jac_min_d(x):
                        return np.stack([
                            mt.layer_ineq_jac_alpha(
                                x, self.jac_fun, min_out_id,
                                s_out.value[min_out_i], s_in),
                            mt.layer_ineq_jac_alpha(
                                x, self.jac_fun, min_in_id,
                                s_in.value[min_in_i], s_out)],
                            axis=0)

                    # Solve
                    sol = mt.alpha_solver(
                        x0=x0[:, i],
                        fun=lambda x: mt.layer_eq_constr_alpha(
                            x, self.prob_fun, s_out, s_in, num_e),
                        jac=lambda x: mt.layer_eq_jac_alpha(
                            x, self.jac_fun, s_out, s_in),
                        min_d=min_d,
                        jac_min_d=jac_min_d,
                        tol=tol,
                        max_iter=max_iter,
                        verbose=verbose,
                        full_return=True)

                elif method == "newton":
                    sol = mt.newton_solver(
                        x0=x0[:, i],
                        fun=lambda x: mt.layer_f_jac(
                            self.prob_fun, self.jac_fun, x,
                            s_out, s_in, num_e),
                        tol=tol,
                        xtol=xtol,
                        max_iter=max_iter,
                        verbose=verbose,
                        full_return=True)

                elif method == "fixed-point":
                    sol = mt.fixed_point_solver(
                        x0=x0[:, i],
                        fun=lambda x: mt.layer_iterative(
                            x, s_out, s_in, num_e),
                        xtol=xtol,
                        max_iter=max_iter,
                        verbose=verbose,
                        full_return=True)

                else:
                    raise ValueError("The selected method is not valid.")

                # Update results and check convergence
                self.param[:, i] = sol.x
                self.solver_output[i] = sol

                if not sol.converged:
                    msg = 'Fit of layer {} '.format(i) + 'did not converge'
                    warnings.warn(msg, UserWarning)

        # Fit with single parameter
        else:
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_labels),
                                  i_col='id', j_col='label',
                                  data_col='value', kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_labels),
                                 i_col='id', j_col='label',
                                 data_col='value', kind='csr')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indptr
            s_in_j = s_in.indices
            s_in_w = s_in.data

            # Init results
            if self.min_degree:
                self.param = np.empty((2, 1), dtype=np.float64)
            else:
                self.param = np.empty((1, 1), dtype=np.float64)
                
            if self.min_degree:
                # Find min degree node
                min_out_i = np.argmin(s_out.value)
                min_in_i = np.argmin(s_in.value)
                if s_out.value[min_out_i] <= s_in.value[min_in_i]:
                    def min_d(x):
                        return mt.fit_ineq_constr_alpha(
                            x, self.prob_fun, min_out_i,
                            s_out.value[min_out_i], s_in)

                    def jac_min_d(x):
                        return mt.fit_ineq_jac_alpha(
                            x, self.jac_fun, min_out_i,
                            s_out.value[min_out_i], s_in)
                else:
                    def min_d(x):
                        return mt.fit_ineq_constr_alpha(
                            x, self.prob_fun, min_in_i,
                            s_in.value[min_in_i], s_out)

                    def jac_min_d(x):
                        return mt.fit_ineq_jac_alpha(
                            x, self.jac_fun, min_in_i,
                            s_in.value[min_in_i], s_out)

                # Solve
                sol = mt.alpha_solver(
                    x0=x0[:, 0],
                    fun=lambda x: mt.fit_eq_constr_alpha(
                        x, self.prob_fun, s_out, s_in, num_e),
                    jac=lambda x: mt.fit_eq_jac_alpha(
                        x, self.jac_fun, s_out, s_in),
                    min_d=min_d,
                    jac_min_d=jac_min_d,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    full_return=True)

            elif method == "newton":
                sol = mt.newton_solver(
                    x0=x0[:, 0],
                    fun=lambda x: mt.stripe_f_jac(
                        self.prob_fun, self.jac_fun, x,
                        s_out_i, s_out_j, s_out_w,
                        s_in_i, s_in_j, s_in_w, self.num_edges),
                    tol=tol,
                    xtol=xtol,
                    max_iter=max_iter,
                    verbose=verbose,
                    full_return=True)

            else:
                raise ValueError("The selected method is not valid.")

            # Update results and check convergence
            self.param = np.array([sol.x])
            self.solver_output = sol

            if not sol.converged:
                msg = 'Fit did not converge.'
                warnings.warn(msg, UserWarning)

    def expected_num_edges(self):
        """ Compute the expected number of edges (per label).
        """
        if not hasattr(self, 'exp_num_edges'):
            if not hasattr(self, 'param'):
                raise Exception('Model must be fitted before hand.')
            
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_labels),
                                  i_col='id', j_col='label',
                                  data_col='value', kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_labels),
                                 i_col='id', j_col='label',
                                 data_col='value', kind='csr')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indptr
            s_in_j = s_in.indices
            s_in_w = s_in.data

            # Compute expected value
            self.exp_num_edges = mt.stripe_exp_edges(
                    self.prob_fun,
                    self.param,
                    s_out_i, s_out_j, s_out_w,
                    s_in_i, s_in_j, s_in_w,
                    self.per_label)

        return self.exp_num_edges
        
    def expected_num_edges_label(self):
        """ Compute the expected number of edges (per label).
        """
        if not hasattr(self, 'exp_num_edges_label'):
            if not hasattr(self, 'param'):
                raise Exception('Model must be fitted before hand.')

            self.exp_num_edges_label = mt.stripe_exp_edges_label(
                        self.prob_fun,
                        self.param,
                        self.out_strength,
                        self.in_strength,
                        self.num_labels,
                        self.per_label)

        return self.exp_num_edges_label
                
    def expected_out_degree(self):
        """ Compute the expected out degree for a given z.
        """

        if not hasattr(self, 'exp_out_degree'):
            if not hasattr(self, 'param'):
                raise Exception('Model must be fitted before hand.')

            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_labels),
                                  i_col='id', j_col='label', data_col='value',
                                  kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_labels),
                                 i_col='id', j_col='label', data_col='value',
                                 kind='csr')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indptr
            s_in_j = s_in.indices
            s_in_w = s_in.data

            # Get out_degree
            self.exp_out_degree, self.exp_in_degree = mt.stripe_exp_degree(
                    self.prob_fun, self.param, s_out_i, s_out_j, s_out_w,
                    s_in_i, s_in_j, s_in_w, self.num_vertices, self.per_label)

        return self.exp_out_degree

    def expected_in_degree(self):
        """ Compute the expected in degree for a given z.
        """
        if not hasattr(self, 'exp_in_degree'):
            _ = self.expected_out_degree()

        return self.exp_in_degree

    def expected_out_degree_by_label(self):
        """ Compute the expected out degree by label for a given z.
        """

        if not hasattr(self, 'exp_out_degree_label'):
            if not hasattr(self, 'param'):
                raise Exception('Model must be fitted before hand.')

            res = mt.stripe_exp_degree_label(
                    self.prob_fun, self.param, self.out_strength,
                    self.in_strength, self.num_labels, self.per_label)

            d_out = np.array(res[0])
            self.exp_out_degree_label = d_out.view(
                type=np.recarray,
                dtype=[('label', 'f8'),
                       ('id', 'f8'),
                       ('value', 'f8')]
                ).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                ).reshape((d_out.shape[0],))

            d_in = np.array(res[1])
            self.exp_in_degree_label = d_in.view(
                type=np.recarray,
                dtype=[('label', 'f8'),
                       ('id', 'f8'),
                       ('value', 'f8')]
                ).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                ).reshape((d_in.shape[0],))

        return self.exp_out_degree_label

    def expected_in_degree_by_label(self):
        """ Compute the expected in degree by label for a given z.
        """
        if not hasattr(self, 'exp_in_degree_label'):
            _ = self.expected_out_degree_by_label()

        return self.exp_in_degree_label

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before sampling.')

        if self.min_degree and not hasattr(self, 'alpha'):
            raise Exception('Ensemble has to be fitted before sampling.')

        # Generate uninitialised graph object
        g = graphs.WeightedLabelGraph.__new__(graphs.WeightedLabelGraph)
        g.lv = graphs.LabelVertexList()

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        g.id_dtype = self.id_dtype
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Sample edges and extract properties
        e = mt.stripe_sample(self.prob_fun, self.param, self.out_strength,
                             self.in_strength, self.num_labels, self.per_label)

        e = e.view(type=np.recarray,
                   dtype=[('label', 'f8'),
                          ('src', 'f8'),
                          ('dst', 'f8'),
                          ('weight', 'f8')]).reshape((e.shape[0],))
        g.num_labels = self.num_labels
        g.label_dtype = self.label_dtype
        e = e.astype([('label', g.label_dtype),
                      ('src', g.id_dtype),
                      ('dst', g.id_dtype),
                      ('weight', 'f8')])
        g.sort_ind = np.argsort(e)
        g.e = e[g.sort_ind]
        g.num_edges = mt.compute_num_edges(g.e)
        ne_label = mt.compute_num_edges_by_label(g.e, g.num_labels)
        dtype = 'u' + str(mt.get_num_bytes(np.max(ne_label)))
        g.num_edges_label = ne_label.astype(dtype)
        g.total_weight = np.sum(e.weight)
        g.total_weight_label = mt.compute_tot_weight_by_label(
                g.e, g.num_labels)

        return g


class BlockFitnessModel(GraphEnsemble):
    """ A generalized fitness model that allows for grouped vertices.

    This model allows to take into account the group of each vertex and
    include this information as part of the model. The strength sequence is
    therefore now subdivided in strength from and to each group.

    The quantity preserved by the ensemble is the total number of edges.

    Attributes
    ----------
    out_strength: np.ndarray
        the out strength sequence
    in_strength: np.ndarray
        the in strength sequence
    num_edges: int
        the total number of edges
    num_vertices: int
        the total number of nodes
    num_groups: int
        the total number of groups by which the vector strengths are computed
    param: np.ndarray
        the parameter vector
    """

    def __init__(self, *args, scale_invariant=False, **kwargs):
        """ Return a BlockFitnessModel for the given graph data.

        The model accepts as arguments either: a DirectedGraph object, the
        strength sequences (in and out) and the number of edges,
        or the strength sequences and the parameter.

        The model accepts the strength sequences as numpy recarrays. The first
        column must contain the node index, the second column the group index
        to which the strength refers, and in the third column must have the
        value of the strength for the node group pair. All node group pairs
        not included are assumed zero.
        """

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.DirectedGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges
                self.num_groups = g.num_groups
                self.group_dict = g.v.group
                self.out_strength = g.out_strength_by_group(get=True)
                self.in_strength = g.in_strength_by_group(get=True)
            else:
                raise ValueError('First argument passed must be a '
                                 'DirectedGraph.')

            if len(args) > 1:
                msg = ('Unnamed arguments other than the Graph have been '
                       'ignored.')
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = ['num_vertices', 'num_edges', 'num_groups',
                             'out_strength', 'in_strength', 'param',
                             'discrete_weights', 'group_dict']
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

        if not hasattr(self, 'num_groups'):
            raise ValueError('Number of groups not set.')
        else:
            try: 
                assert self.num_groups / int(self.num_groups) == 1
                self.num_groups = int(self.num_groups)
            except Exception:
                raise ValueError('Number of groups must be an integer.')

            if self.num_groups <= 0:
                raise ValueError(
                    'Number of groups must be a positive number.')

        if not hasattr(self, 'group_dict'):
            raise ValueError('Group dictionary not set.')
        else:
            if isinstance(self.group_dict, dict):
                self.group_dict = mt.dict_to_array(self.group_dict)
            elif isinstance(self.group_dict, np.ndarray):
                msg = 'Group_dict must have one element for each vertex.'
                assert len(self.group_dict) == self.num_vertices
            else:
                ValueError('Group dictionary must be a dict or an array.')

        if not hasattr(self, 'out_strength'):
            raise ValueError('out_strength not set.')

        if not hasattr(self, 'in_strength'):
            raise ValueError('in_strength not set.')

        if not (hasattr(self, 'num_edges') or
                hasattr(self, 'param')):
            raise ValueError('Either num_edges or param must be set.')

        # Ensure that strengths passed adhere to format
        msg = ("Out strength must be a rec array with columns: "
               "('id', 'group', 'value')")
        assert isinstance(self.out_strength, np.recarray), msg
        for clm in self.out_strength.dtype.names:
            assert clm in ('id', 'group', 'value'), msg

        msg = ("In strength must be a rec array with columns: "
               "('id', 'group', 'value')")
        assert isinstance(self.in_strength, np.recarray), msg
        for clm in self.in_strength.dtype.names:
            assert clm in ('id', 'group', 'value'), msg

        # Ensure that num_vertices and num_groups are coherent with info
        # in strengths
        if self.num_vertices < max(np.max(self.out_strength.id),
                                   np.max(self.in_strength.id)):
            raise ValueError(
                'Number of vertices smaller than max id value in strengths.')

        if self.num_groups < max(np.max(self.out_strength.group),
                                 np.max(self.in_strength.group)):
            raise ValueError(
                'Number of labels smaller than max group value in strengths.')

        # Ensure that all groups, ids and values of the strength are positive
        if np.any(self.out_strength.group < 0):
            raise ValueError(
                "Out strength groups must contain positive values only.")

        if np.any(self.in_strength.group < 0):
            raise ValueError(
                "In strength groups must contain positive values only.")

        if np.any(self.out_strength.id < 0):
            raise ValueError(
                "Out strength ids must contain positive values only.")

        if np.any(self.in_strength.id < 0):
            raise ValueError(
                "In strength ids must contain positive values only.")

        if np.any(self.out_strength.value < 0):
            raise ValueError(
                "Out strength values must contain positive values only.")

        if np.any(self.in_strength.value < 0):
            raise ValueError(
                "In strength values must contain positive values only.")

        msg = "Storing zeros in the strengths leads to inefficient code."
        if np.any(self.out_strength.value == 0) or np.any(
                self.in_strength.value == 0):
            msg = "Storing zeros in the strengths leads to inefficient code."
            warnings.warn(msg, UserWarning)

        # Ensure that strengths are sorted
        self.out_strength = self.out_strength[['id', 'group', 'value']]
        self.in_strength = self.in_strength[['id', 'group', 'value']]
        self.out_strength.sort()
        self.in_strength.sort()

        # Ensure that the parameters or number of edges are set correctly
        if hasattr(self, 'num_edges'):
            if not isinstance(self.num_edges, np.ndarray):
                self.num_edges = np.array([self.num_edges], dtype=np.float64)

            msg = ('Number of edges must be a number.')
            if len(self.num_edges) > 1:
                raise ValueError(msg)

            if not np.issubdtype(self.num_edges.dtype, np.number):
                raise ValueError(msg)

            if np.any(self.num_edges < 0):
                msg = 'Parameter must be positive.'
                raise ValueError(msg)
        else:
            if not isinstance(self.param, np.ndarray):
                self.param = np.array([self.param])

            msg = ('Parameter must be a number.')
            if len(self.param) > 1:
                raise ValueError(msg)

            if not np.issubdtype(self.param.dtype, np.number):
                raise ValueError(msg)

            if np.any(self.param < 0):
                msg = 'Parameter must be positive.'
                raise ValueError(msg)

        # Check that sum of in and out strengths are equal
        tot_out = np.sum(self.out_strength.value)
        tot_in = np.sum(self.in_strength.value)

        msg = 'Sum of strengths do not match.'
        assert np.allclose(tot_out, tot_in, atol=1e-14, rtol=1e-9), msg

        # Get the correct probability functional
        self.scale_invariant = scale_invariant
        if scale_invariant:
            self.prob_fun = mt.p_invariant
            self.jac_fun = mt.jac_invariant
        else:
            self.prob_fun = mt.p_fitness
            self.jac_fun = mt.jac_fitness

        # If param is set computed expected number of edges per label
        if hasattr(self, 'param'):
            self.num_edges = self.expected_num_edges()

    def fit(self, x0=None, method=None, tol=1e-5,
            xtol=1e-12, max_iter=100, verbose=False):
        """ Compute the optimal z to match the given number of edges.

        Parameters
        ----------
        x0: np.ndarray
            initial conditions for parameters
        method: 'newton' or 'fixed-point'
            selects which method to use for the solver
        tol: float
            tolerance for the exit condition on the norm
        xtol: float
            tolerance for the exit condition on difference between two
            iterations
        max_iter: int or float
            maximum number of iteration
        verbose: boolean
            if true print debug info while iterating

        """
        if method is None:
            method = 'newton'

        if (method == 'fixed-point') and self.scale_invariant:
            raise Exception('Fixed point solver not supported for scale '
                            'invariant functional.')

        if x0 is None:
            x0 = np.zeros(1, dtype=np.float64)
        else:
            if not isinstance(x0, np.ndarray):
                x0 = np.array([x0])

            try:
                x0 = np.float64(self.z)
            except TypeError:
                raise TypeError('x0 must be a float.')

            if len(x0) > 1:
                raise ValueError('x0 must be a single number.')

        if len(self.num_edges) == 1:
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_groups),
                                  kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_groups),
                                 kind='csc')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indices
            s_in_j = s_in.indptr
            s_in_w = s_in.data

            if method == "newton":
                sol = mt.newton_solver(
                    x0=x0,
                    fun=lambda x: mt.block_f_jac(
                        self.prob_fun, self.jac_fun, x, s_out_i, s_out_j,
                        s_out_w, s_in_i, s_in_j, s_in_w, self.group_dict,
                        self.num_edges),
                    tol=tol,
                    xtol=xtol,
                    max_iter=max_iter,
                    verbose=verbose,
                    full_return=True)
            elif method == "fixed-point":
                sol = mt.fixed_point_solver(
                    x0=x0,
                    fun=lambda x: mt.block_iterative(
                        x, s_out_i, s_out_j, s_out_w, s_in_i, s_in_j, s_in_w,
                        self.group_dict, self.num_edges),
                    xtol=xtol,
                    max_iter=max_iter,
                    verbose=verbose,
                    full_return=True)

            else:
                raise ValueError("The selected method is not valid.")

            # Update results and check convergence
            self.param = sol.x
            self.solver_output = sol

            if not sol.converged:
                if method == 'newton':
                    mod = sol.norm_seq[-1]
                else:
                    mod = self.expected_num_edges() - self.num_edges
                if sol.max_iter_reached:
                    msg = ('Fit did not converge: \n solver stopped because'
                           ' it reached the max number of iterations. \n'
                           'Final distance from root = {}'.format(mod))
                    warnings.warn(msg, UserWarning)

                if method == 'newton':
                    if sol.no_change_stop:
                        msg = ('Fit did not converge: \n solver stopped '
                               'because the update of x was smaller than the '
                               ' tolerance. \n Final distance from'
                               ' root = {}'.format(mod))
                        warnings.warn(msg, UserWarning)

        else:
            raise ValueError('Number of edges must be a single value.')

    def expected_num_edges(self):
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before hand.')

        if not hasattr(self, 'exp_num_edges'):
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_groups),
                                  kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_groups),
                                 kind='csc')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indices
            s_in_j = s_in.indptr
            s_in_w = s_in.data

            self.exp_num_edges = mt.block_exp_num_edges(
                self.prob_fun, self.param, s_out_i, s_out_j, s_out_w,
                s_in_i, s_in_j, s_in_w, self.group_dict)

        return self.exp_num_edges

    def expected_out_degree(self):
        """ Compute the expected out degree for a given z.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before hand.')

        if not hasattr(self, 'exp_out_degree'):
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_groups),
                                  kind='csr')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_groups),
                                 kind='csc')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indptr
            s_out_j = s_out.indices
            s_out_w = s_out.data
            s_in_i = s_in.indices
            s_in_j = s_in.indptr
            s_in_w = s_in.data

            # Get out_degree
            self.exp_out_degree = mt.block_exp_out_degree(
                self.prob_fun, self.param, s_out_i, s_out_j, s_out_w, s_in_i,
                s_in_j, s_in_w, self.group_dict)

        return self.exp_out_degree

    def expected_in_degree(self):
        """ Compute the expected in degree for a given z.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before hand.')

        if not hasattr(self, 'exp_in_degree'):
            # Convert to sparse matrices
            s_out = lib.to_sparse(self.out_strength,
                                  (self.num_vertices, self.num_groups),
                                  kind='csc')
            s_in = lib.to_sparse(self.in_strength,
                                 (self.num_vertices, self.num_groups),
                                 kind='csr')

            if not s_out.has_sorted_indices:
                s_out.sort_indices()
            if not s_in.has_sorted_indices:
                s_in.sort_indices()

            # Extract arrays from sparse matrices
            s_out_i = s_out.indices
            s_out_j = s_out.indptr
            s_out_w = s_out.data
            s_in_i = s_in.indptr
            s_in_j = s_in.indices
            s_in_w = s_in.data

            # Get in_degree (note switched positions of args)
            self.exp_in_degree = mt.block_exp_out_degree(
                self.prob_fun, self.param, s_in_i, s_in_j, s_in_w, s_out_i,
                s_out_j, s_out_w, self.group_dict)

        return self.exp_in_degree

    def sample(self):
        """ Return a Graph sampled from the ensemble.
        """
        if not hasattr(self, 'param'):
            raise Exception('Ensemble has to be fitted before hand.')

        # Generate uninitialised graph object
        g = graphs.WeightedGraph.__new__(graphs.WeightedGraph)
        g.gv = graphs.GroupVertexList()
        g.num_groups = len(np.unique(self.group_dict))

        # Initialise common object attributes
        g.num_vertices = self.num_vertices
        num_bytes = mt.get_num_bytes(self.num_vertices)
        g.id_dtype = np.dtype('u' + str(num_bytes))
        g.v = np.arange(g.num_vertices, dtype=g.id_dtype).view(
            type=np.recarray, dtype=[('id', g.id_dtype)])
        g.id_dict = {}
        for x in g.v.id:
            g.id_dict[x] = x

        # Convert to sparse matrices
        s_out = lib.to_sparse(self.out_strength,
                              (self.num_vertices, self.num_groups),
                              kind='csr')
        s_in = lib.to_sparse(self.in_strength,
                             (self.num_vertices, self.num_groups),
                             kind='csc')

        if not s_out.has_sorted_indices:
            s_out.sort_indices()
        if not s_in.has_sorted_indices:
            s_in.sort_indices()

        # Extract arrays from sparse matrices
        s_out_i = s_out.indptr
        s_out_j = s_out.indices
        s_out_w = s_out.data
        s_in_i = s_in.indices
        s_in_j = s_in.indptr
        s_in_w = s_in.data

        # Sample edges and extract properties
        e = mt.block_sample(
            self.prob_fun, self.param, s_out_i, s_out_j, s_out_w,
            s_in_i, s_in_j, s_in_w, self.group_dict)
        e = e.view(type=np.recarray,
                   dtype=[('src', 'f8'),
                          ('dst', 'f8'),
                          ('weight', 'f8')]).reshape((e.shape[0],))
        e = e.astype([('src', g.id_dtype),
                      ('dst', g.id_dtype),
                      ('weight', 'f8')])
        g.sort_ind = np.argsort(e)
        g.e = e[g.sort_ind]
        g.num_edges = mt.compute_num_edges(g.e)
        g.total_weight = np.sum(e.weight)

        return g
