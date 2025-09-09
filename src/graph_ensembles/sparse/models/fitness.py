from .ensembles import DiGraphEnsemble
from .ensembles import MultiDiGraphEnsemble
from .ensembles import empty_index
from .solver import monotonic_newton_solver
import warnings
import scipy.sparse as sp
from . import graphs
import numpy as np
from numba import jit
from numba import njit, prange
from math import isinf
from numba.typed import List
from math import log
from math import log1p
from ..graphs import common_functions


class FitnessModel(DiGraphEnsemble, common_functions):
    """The Fitness model takes the fitnesses of each node in order to
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
        """Return a FitnessModel for the given graph data.

        The model accepts as arguments either: a DiGraph, in which case the
        strengths are used as fitnesses, or directly the fitness sequences (in
        and out). The model accepts the fitness sequences as numpy arrays.
        """

        super().__init__(*args, **kwargs)
        self.kind = 'exp'

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.DiGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_edges = g.num_edges()
                self.prop_out = g.out_strength()
                self.prop_in = g.in_strength()
                self.intra_size = g.get("intra_size")

                self.level = g.level
                self.__dict__.update(kwargs)
                
                # force attributes wrt intra_size
                if g.intra_size < 1:
                    self.vsplit = g.vsplit
            else:
                raise ValueError("First argument passed must be a " "DiGraph.")

            if len(args) > 1:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)
        
        elif len(kwargs) > 0:
            self.__dict__.update(kwargs)
        
        self.fit_method_title = self.fit_method[len("num_edges_"):].title()

        self._create_vars_dir()

        # Get options from keyword arguments
        # allowed_arguments = [
        #     "num_vertices",
        #     "num_edges",
        #     "prop_out",
        #     "prop_in",
        #     "param",
        #     "selfloops",
        #     "name",
        #     "level",
        #     "vsplit",
        #     "intra_size",
        #     "full_intra_row",
        #     "fit_method",
        #     "corpkey",
        # ]
        # for name in kwargs:
        #     if name not in allowed_arguments:
        #         raise ValueError("Illegal argument passed: " + name)
        #     else:
        #         setattr(self, name, kwargs[name])
                

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
        
        if not hasattr(self, "selfloops"):
            self.selfloops = False

        # if not hasattr(self, "prop_out"):
        #     raise ValueError("prop_out not set.")
        # elif isinstance(self.prop_out, empty_index):
        #     raise ValueError("prop_out not set.")

        # if not hasattr(self, "prop_in"):
        #     raise ValueError("prop_in not set.")
        # elif isinstance(self.prop_in, empty_index):
        #     raise ValueError("prop_in not set.")


        # # Ensure that fitnesses passed adhere to format (ndarray)
        # msg = "Node out properties must be a numpy array of length " + str(
        #     self.num_vertices
        # )
        # assert isinstance(self.prop_out, np.ndarray), msg
        # assert self.prop_out.shape == (self.num_vertices,), msg

        # msg = "Node in properties must be a numpy array of length " + str(
        #     self.num_vertices
        # )
        # assert isinstance(self.prop_in, np.ndarray), msg
        # assert self.prop_in.shape == (self.num_vertices,), msg

        # # Ensure that fitnesses have positive values only
        # msg = "Node out properties must contain positive values only."
        # assert np.all(self.prop_out >= 0), msg

        # msg = "Node in properties must contain positive values only."
        # assert np.all(self.prop_in >= 0), msg

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

    def delete_ensemble_samples(self, corpkey):
        if corpkey:
            import os, shutil
            samples_folder = self.vars_dir_ensembles + "/samples"
            if os.path.exists(samples_folder):
                print(f'===\nDeleting the folder: {samples_folder}\n===\n',)
                shutil.rmtree(samples_folder)
    
    def set_num_vertices_out_in_strengths_to(self, g):
        """
        Set some variables to observed graph g
        """
        
        self.num_vertices = g.num_vertices
        self.prop_out = g.out_strength()
        self.prop_in = g.in_strength()

    def mean_std_sampled_graphs(self, ivec_name, num_sampled_graphs):
        """
        Compute entry-wise running mean for a given meas over sampled graphs.
        Returns the mean array.
        """
        from ...utils import load_dict
        from os import path
        
        # load it
        full_path = self.vars_dir_ensembles + f"/{ivec_name}/num_samples_{num_sampled_graphs}.csv"
        if path.exists(full_path):
            return np.genfromtxt(full_path) 

        # create the meas and save it
        if num_sampled_graphs == 0:
            ivec, ivec_std = np.zeros(self.num_vertices*2, dtype = float).reshape(2, self.num_vertices)
        else:
            test_ivec = load_dict(self.vars_dir_ensembles + f"/samples/graph0/graph0.pkl")[ivec_name]
            
            # create ivec and ivec_std as array
            ivec = np.zeros_like(np.atleast_1d(test_ivec), dtype = float)
            ivec_std = ivec.copy()

            # recursive mean and std
            for i in range(num_sampled_graphs):
                gi_val = load_dict(self.vars_dir_ensembles + f"/samples/graph{i}/graph{i}.pkl")[ivec_name]
                ivec, ivec_std = self.recursive_mean_std(i, ivec, ivec_std, gi_val)

            # save the mean and std as [[mean],[std]]
            # np.savetxt(full_path, X = np.vstack((ivec, ivec_std)))
            
        return ivec, ivec_std

    def recursive_mean_std(self, i, ens_mean=None, ens_std=None, new_meas=None):
        """
        Compute the recursive mean and standard deviation for a given set of measurements.
        Algorthm stable "Welford's online algorithm" 
        [https://doi.org/10.1080/00401706.1962.10490022]
        [https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/]

        Parameters:
        - i (int): Current index (number of measurements processed so far).
        - ens_mean (np.ndarray): Current mean values (shape: (N,)).
        - ens_std (np.ndarray): Current standard deviation values (shape: (N,)).
        - new_meas (np.ndarray): New measurement values to include (shape: (N,)).

        Returns:
        - ens_mean (np.ndarray): Updated mean values.
        - ens_std (np.ndarray): Updated standard deviation values.
        """
        # Update mean
        N = i + 1
        
        # if i = 0, ens_mean = [0, 0, ...] updates while the std stays zero
        delta = new_meas - ens_mean
        new_mean = ens_mean + delta / N  # using += that is in-place and changes also the outer ens_mean

        # --- Update Standard Error of the Mean (SEM) ---
        # Just s^2_N <--> N * sem^2_N, in the formulas above
        # sem^2_N = (N-2) * (N-1) * sem_{N-1} + (x_N - bar(x)_N) * (x_N - bar(x)_{N-1}) / [(N-1) * N]
        if N > 1:
            sem_old_squared = ens_std**2
            s_old_squared = sem_old_squared * (N - 1)
            
            new_delta = new_meas - new_mean
            
            # Calculate the new sample variance
            s_new_squared = ((N - 2) * s_old_squared + delta * new_delta) / (N - 1)
            
            # Calculate the new SEM from the new sample variance
            # SEM_new = sqrt(s_new^2 / N)
            new_sem = np.sqrt(s_new_squared / N)
        else:
            # SEM is 0 for a single data point.
            new_sem = np.zeros_like(new_mean)
            
        return new_mean, new_sem

    def set_ensemble_variables(self, meas_name = ["ivec"], num_samples = 1):
        
        import os
        from ...utils import max_sampled_graph_idx

        # Create the ensemble of sampled nets
        self.vars_dir_ensembles = self.vars_dir.replace("vars", "vars/ensembles")
        
        # find the max graph idx in the model.vars_dir and sample the rest
        num_sampled_graphs = max_sampled_graph_idx(self.vars_dir_ensembles + "/samples", num_samples) + 1
        
        # save norm of differences among two ensemble mean
        for m in meas_name:
            self.__dict__[m], self.__dict__[m+"_std"] = self.mean_std_sampled_graphs(m, num_sampled_graphs)
            os.makedirs(self.vars_dir_ensembles + f"/{m}", exist_ok = True)

        return num_sampled_graphs

    def load_or_fit(
        self,
        x0=None,
        method="num_edges",
        atol=1e-24,
        rtol=1e-9,
        maxiter=100,
        verbose=False,):
        import os
        
        # load the invariant model on the gI or fit it
        path_param = self.vars_dir + "/param.csv"
        
        if os.path.exists(path_param):
            
            from graph_ensembles.utils import load_array
            self.param = np.expand_dims(load_array(path_param), axis = 0) # The code needs np.array([#])
            print(f'-Load the parameter enforcing {self.fit_method} -> param: {self.param}',)
            
        else:
            print(f'-Fit the parameter with {self.fit_method}',)
            self.fit(
                x0=x0,
                method=method,
                atol=atol,
                rtol=rtol,
                maxiter=maxiter,
                verbose=verbose,
                )
                
            os.makedirs(os.path.dirname(path_param), exist_ok = True)
            np.savetxt(path_param, self.param, delimiter = ",")
        


    def fit(
        self,
        x0=None,
        method="num_edges",
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
        method: 'num_edges' or 'mle'
            Selects whether to fit param using maximum likelihood estimation
            or by ensuring that the expected num_edges matches the given one.
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

        if method == "num_edges":
            # Ensure that num_edges is set
            if not hasattr(self, "num_edges"):
                raise ValueError("Number of edges must be set for num_edges solver.")
            sol = monotonic_newton_solver(
                x0,
                self.num_edges_fit_fun,
                self.num_edges,
                atol=atol,
                rtol=rtol,
                x_l=0.0,
                x_u=np.inf,
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

    def num_edges_fit_fun(self, delta):
        """Return the objective function value and the Jacobian
        for a given value of delta.
        """
        f, jac = self.exp_edges_f_jac(
            self.num_edges_jac_i,
            delta,
            self.prop_out_I,
            self.prop_in_I,
            self.prop_out_R,
            self.prop_in_R,
            self.prop_dyad,
            self.selfloops,
            self.fit_method,
        )
        
        return f, jac
    
    @staticmethod
    @njit(parallel=True)
    def exp_edges_f_jac(p_jac_ij, param, prop_out, prop_in, prop_dyad, selfloops):
        """
        Compute the objective function of the num_edges solver and its
        derivative.
        """
        N = len(prop_out)

        # Preallocate result vectors for each outer loop iteration (i)
        # These arrays store intermediate totals per i, which can be summed later
        f_vector = np.zeros(N)
        jac_vector = np.zeros(N)

        # Outer loop is parallelized with prange
        # This is the correct and efficient use of numba's parallelism
        for i in prange(N):
            p_out_i = prop_out[i]

            # Use scalar accumulators for better memory efficiency and cache usage
            f_i = 0.0
            jac_i = 0.0

            # Inner loop is serial — this is good because nested prange is not well supported
            for j in range(N):
                if (i != j) or selfloops:
                    p_in_j = prop_in[j]

                    # Call the user-defined function to get value and jacobian for dyad (i, j)
                    p_val, jac_val = p_jac_ij(param, p_out_i, p_in_j, prop_dyad(i,j))

                    # Accumulate results efficiently in scalars
                    f_i += p_val
                    jac_i += jac_val

            # Store per-node results
            f_vector[i] = f_i
            jac_vector[i] = jac_i

        # Sum across all nodes to get final result (parallel reduction is fast for large n)
        f_vector, jac_vector = np.sum(f_vector), np.sum(jac_vector)

        return f_vector, jac_vector

    @staticmethod
    def num_edges_fit(num_edges_int, num_edges_bet, fit_method):
        """Calculate the number of edges to be enforced"""
        
        _num_edges_fit = 0
        if "intra" in fit_method:
            _num_edges_fit += num_edges_int
        
        if "bet" in fit_method:
            _num_edges_fit += num_edges_bet
        return _num_edges_fit



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
            return tmp1 / (1 + tmp1), tmp / (1 + tmp1) ** 2

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
            return tmp / (1 + tmp)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def logp(d, x_i, y_j, z_ij):
        """Compute the log probability of connection between node i and j."""
        if (x_i == 0) or (y_j == 0) or (z_ij == 0) or (d[0] == 0):
            return -np.inf

        tmp = d[0] * x_i * y_j * z_ij
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
        if (x_i == 0) or (y_j == 0) or (z_ij == 0) or (d[0] == 0):
            return 0.0

        tmp = d[0] * x_i * y_j * z_ij
        if isinf(tmp):
            return -np.inf
        else:
            return log1p(-tmp / (1 + tmp))


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

        # If an argument is passed then it must be a graph
        if len(args) > 0:
            if isinstance(args[0], graphs.MultiDiGraph):
                g = args[0]
                self.num_vertices = g.num_vertices
                self.num_labels = g.num_labels
                self.num_edges = g.num_edges()
                self.num_edges_label = g.num_edges_label()
                self.prop_out = g.out_strength_by_label()
                self.prop_in = g.in_strength_by_label()
                self.per_label = True
            else:
                raise ValueError("First argument passed must be a " "MultiDiGraph.")

            if len(args) > 1:
                msg = "Unnamed arguments other than the Graph have been " "ignored."
                warnings.warn(msg, UserWarning)

        # Get options from keyword arguments
        allowed_arguments = [
            "num_vertices",
            "num_edges",
            "num_edges_label",
            "num_labels",
            "prop_out",
            "prop_in",
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
        self.prop_out = sp.csr_array(self.prop_out)
        self.prop_in = sp.csr_array(self.prop_in)

        # Ensure that all fitness are positive
        if np.any(self.prop_out.data < 0):
            raise ValueError("Node out properties must contain positive values only.")

        if np.any(self.prop_in.data < 0):
            raise ValueError("Node in properties must contain positive values only.")

        # Convert to list of tuples
        self.prop_out = self.csx_to_tuple_list(
            self.prop_out.indptr, self.prop_out.indices, self.prop_out.data
        )
        self.prop_in = self.csx_to_tuple_list(
            self.prop_in.indptr, self.prop_in.indices, self.prop_in.data
        )

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

    def fit(
        self,
        x0=None,
        method="num_edges",
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
        method: 'num_edges' or 'mle'
            Selects whether to fit param using maximum likelihood estimation
            or by ensuring that the expected num_edges matches the given one.
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

        if method == "num_edges":
            if self.per_label:
                # Ensure that num_edges is set
                if not hasattr(self, "num_edges_label"):
                    raise ValueError(
                        "Number of edges per label must be set for num_edges "
                        "solver with per_label option."
                    )

                self.solver_output = [None] * self.num_labels
                self.param = x0.copy()

                # Transform properties to csc format to select labels quickly
                prop_out = sp.csr_array(
                    self.tuple_list_to_csx(self.prop_out),
                    shape=(self.num_vertices, self.num_labels),
                ).tocsc()
                prop_out = self.csx_to_tuple_list(
                    prop_out.indptr, prop_out.indices, prop_out.data
                )
                prop_in = sp.csr_array(
                    self.tuple_list_to_csx(self.prop_in),
                    shape=(self.num_vertices, self.num_labels),
                ).tocsc()
                prop_in = self.csx_to_tuple_list(
                    prop_in.indptr, prop_in.indices, prop_in.data
                )

                for i in range(self.num_labels):
                    p_out = prop_out[i]
                    p_in = prop_in[i]
                    num_e = self.num_edges_label[i]

                    sol = monotonic_newton_solver(
                        np.array([x0[i]]),
                        lambda x: self.num_edges_fit_layer(x[0], p_out, p_in),
                        num_e,
                        atol=atol,
                        rtol=rtol,
                        x_l=0.0,
                        x_u=np.inf,
                        max_iter=maxiter,
                        full_return=True,
                        verbose=verbose,
                    )

                    # Update results and check convergence
                    self.param[i] = sol.x[0]
                    self.solver_output[i] = sol

                    if not sol.converged:
                        msg = "Fit of layer {} ".format(i) + "did not converge."
                        warnings.warn(msg, UserWarning)
            else:
                # Ensure that num_edges is set
                if not hasattr(self, "num_edges"):
                    raise ValueError("Number of edges must be set for num_edges solver.")

                sol = monotonic_newton_solver(
                    x0[0:1],
                    self.num_edges_fit_fun,
                    self.num_edges,
                    atol=atol,
                    rtol=rtol,
                    x_l=0.0,
                    x_u=np.inf,
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

    def num_edges_fit_fun(self, delta):
        """Return the objective function value and the Jacobian
        for a given value of delta.
        """
        f, jac = self.exp_edges_f_jac(
            self.p_jac_ij, delta, self.prop_out, self.prop_in, self.selfloops
        )

        return f, jac

    def num_edges_fit_layer(self, delta, prop_out, prop_in):
        """Return the objective function value and the Jacobian
        for a given value of delta.
        """
        f, jac = self.exp_edges_f_jac_layer(
            self.p_jac_ijk, delta, prop_out, prop_in, self.selfloops
        )

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac(p_jac_ij, param, prop_out, prop_in, selfloops):
        """Compute the objective function of the num_edges solver and its
        derivative.
        """
        f = 0.0
        jac = 0.0
        for i, p_out_i in enumerate(prop_out):
            for j, p_in_j in enumerate(prop_in):
                if (i != j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ij(param, p_out_i, p_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_f_jac_layer(p_jac_ijk, param, prop_out, prop_in, selfloops):
        """Compute the objective function of the num_edges solver and its
        derivative.
        """
        f = np.float64(0.0)
        jac = np.float64(0.0)
        for i, p_out_i in zip(prop_out[0], prop_out[1]):
            for j, p_in_j in zip(prop_in[0], prop_in[1]):
                if (i != j) | selfloops:
                    p_tmp, jac_tmp = p_jac_ijk(param, p_out_i, p_in_j)
                    f += p_tmp
                    jac += jac_tmp

        return f, jac

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def p_jac_ij(d, prop_out, prop_in):
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
        x_lbl = prop_out[0]
        x_val = prop_out[1]
        y_lbl = prop_in[0]
        y_val = prop_in[1]
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
    def csx_to_tuple_list(indptr, indices, data):
        N = len(indptr) - 1
        res = List()
        for i in range(N):
            m = indptr[i]
            n = indptr[i + 1]
            res.append((indices[m:n], data[m:n]))
        return res

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def tuple_list_to_csx(data):
        # Get nnz elements
        nnz = 0
        for x, _ in data:
            nnz += len(x)

        ind_type = data[0][0].dtype
        val_type = data[0][1].dtype
        indptr = np.zeros(len(data) + 1, dtype=ind_type)
        indices = np.zeros(nnz, dtype=ind_type)
        vals = np.zeros(nnz, dtype=val_type)
        n = 0
        for i, (x, y) in enumerate(data):
            m = n
            n += len(x)
            indptr[i + 1] = n
            indices[m:n] = x
            vals[m:n] = y

        return vals, indices, indptr

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
        val = 1.0

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
    def logp(d, prop_out, prop_in, prop_dyad):
        """Compute the log probability of connection between node i and j."""
        # Initialize result
        i = 0
        j = 0
        val = 1.0

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
                        val /= 1 + tmp
                i += 1
                j += 1
            elif x_lbl[i] < y_lbl[j]:
                i += 1
            else:
                j += 1

        if val == 1.0:
            return -np.inf
        else:
            return log1p(-val)

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def log1mp(d, prop_out, prop_in, prop_dyad):
        """Compute the log of 1 minus the probability of connection between
        node i and j.
        """
        # Initialize result
        i = 0
        j = 0
        val = 1.0

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
                        return -np.inf
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
            return -np.inf

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
            return -np.inf
        else:
            return log1p(-tmp / (1 + tmp))
