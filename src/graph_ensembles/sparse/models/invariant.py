from .fitness import FitnessModel
from .fitness import MultiFitnessModel
import numpy as np
from numba import jit, njit
from math import isinf
from math import log
from math import expm1
from math import exp
from numba import njit, prange
from ... import utils


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

    # @staticmethod
    # def initialize_model(g, gI, vR, kwargs, num_edges_bet = None, ):

    #     fit_method = kwargs["fit_method"]

    #     if kwargs["intra_size"] == 1:
    #         model = sp.ScaleInvariantModel(g, **kwargs)

    #     else:
    #         vR_nodes = vR.T.values[0]
    #         out_stre = lambda i: g.out_strength()[g.id_dict[i]]
    #         in_stre = lambda i: g.in_strength()[g.id_dict[i]]
    #         cmap = lambda stre, nodes: np.array(list(map(stre, nodes)))
    #         kwargs.update({
    #                         "prop_out_R" : cmap(out_stre, vR_nodes), "prop_in_R" : cmap(in_stre, vR_nodes), 
    #                         "prop_out_I" : gI.out_strength(), "prop_in_I" : gI.in_strength(),
    #                         "num_vertices" : len(g.num_vertices) if "bet" in fit_method else len(gI.num_vertices),
    #                         "num_edges" : sp.ScaleInvariantModel.num_edges_fit(gI.num_edges(), num_edges_bet, fit_method),
    #                         "level" : g.level, "intra_size" : gI.intra_size, "vsplit" : gI.vsplit,
    #                         })
    #         model = sp.ScaleInvariantModel(**kwargs)

    #     return model
    @staticmethod
    def initialize_model(g, gI, vR, kwargs):
        """
        Initialize the ScaleInvariantModel based on the provided graph data.

        Parameters:
        -----------
        g : DiGraph
            The full graph.
        gI : DiGraph
            The internal graph (subset of g).
        vR : DataFrame
            External nodes (ROW nodes).
        kwargs : dict
            Additional parameters for the model.

        Returns:
        --------
        ScaleInvariantModel
            The initialized model.
        """
        # Extract the fit method from kwargs
        fit_method = kwargs.get("fit_method", None)
        if fit_method is None:
            raise ValueError("The 'fit_method' parameter is required in kwargs.")

        # If intra_size is 1, initialize the model directly with the full graph
        if kwargs.get("intra_size", 0) == 1:
            model = ScaleInvariantModel(g, **kwargs)
        else:

            # if only "intra" in fit_method
            kwargs.update({
                            "num_vertices": gI.num_vertices,
                            "level": g.level,
                            })
            
            # if there is also "inbetween" update the prop_R, num_vertices and num_edges
            if "bet" in fit_method:

                # extract vR identifier
                vR_nodes = vR.T.values[0]

                # create maps from identifiers to index (g.id_dict[i]) and search for the relative strengths
                out_stre = lambda i: g.out_strength()[g.id_dict[i]]
                in_stre = lambda i: g.in_strength()[g.id_dict[i]]
                cmap = lambda stre, nodes: np.array(list(map(stre, nodes)))

                # Update kwargs with calculated properties
                kwargs.update({
                    "prop_out_R": cmap(out_stre, vR_nodes),
                    "prop_in_R": cmap(in_stre, vR_nodes),
                    "num_vertices": g.num_vertices, # gI.num_vertices + ROW = g.num_vertices
                    })

            # Initialize the model with the updated kwargs
            model = ScaleInvariantModel(**kwargs)

        return model

    @staticmethod
    @njit()
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
    @njit()  # pragma: no cover
    def p_ij(d, x_i, y_j, z_ij = 1.0):
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
            return -np.inf

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
            return -np.inf
        else:
            return -tmp

    @staticmethod
    @njit()
    def num_edges_jac_i(fun, jac, d, x_i, y_j, z_ij = 1.0):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        tmp = x_i * y_j * z_ij
        tmp1 = d[0] * tmp
        
        # vars are immutable so return the updated value
        # print(f'-p_iI: {- np.expm1(-tmp1)}',)
        return fun - np.expm1(-tmp1).sum(), jac + (tmp * np.exp(-tmp1)).sum()
        

    @staticmethod
    @njit(parallel=True)
    def exp_edges_f_jac(num_edges_jac_i, param, prop_out_I, prop_in_I, prop_out_R, prop_in_R, prop_dyad, selfloops, fit_method):

        N = len(prop_out_I)

        # Preallocate result vectors for each outer loop iteration (i)
        # These arrays store intermediate totals per i, which will be summed later
        f_vector = np.zeros(N)
        jac_vector = np.zeros(N)

        # Outer loop is parallelized with prange
        # This is the correct and efficient use of numba's parallelism
        for i in prange(N):
            
            # Use scalar accumulators for better memory efficiency and cache usage
            f_i = 0.0
            jac_i = 0.0
            
            prop_out_i, prop_in_i = prop_out_I[i], prop_in_I[i]
            
            if "intra" in fit_method:
                # print(f'\n-i: {i}',)
                f_i, jac_i = num_edges_jac_i(f_i, jac_i, param, prop_out_i, prop_in_I)
                # print(f'-f_i: {f_i}',)
                f_i, jac_i = num_edges_jac_i(f_i, jac_i, param, prop_out_I, prop_in_i)

                f_i /= 2
                jac_i /= 2
                # print(f'-f_i: {f_i}',)

            if "bet" in fit_method:
                # print(f'-bet',)
                f_i, jac_i = num_edges_jac_i(f_i, jac_i, param, prop_out_i, prop_in_R)
                f_i, jac_i = num_edges_jac_i(f_i, jac_i, param, prop_out_R, prop_in_i)

            # Store per-node results
            f_vector[i] = f_i
            jac_vector[i] = jac_i
            # print(f'-f_vector: {f_vector}',)

        # Sum across all nodes to get final result (parallel reduction is fast for large n)
        f_vector, jac_vector = np.sum(f_vector), np.sum(jac_vector)

        if "intra" in fit_method and not selfloops:
            # discard self-loops
            # print(f'-\n SelfLoops',)
            f_vector, jac_vector = num_edges_jac_i(-f_vector, -jac_vector, param, prop_out_I, prop_in_I)
            
            f_vector *= -1
            jac_vector *= -1

        return f_vector, jac_vector

    @staticmethod
    @njit()
    def num_edges_i(fun, d, x_i, y_j, z_ij = 1.0):
        """Compute the probability of connection and the jacobian
        contribution of node i and j.
        """
        tmp = x_i * y_j * z_ij
        tmp1 = d[0] * tmp
        
        # vars are immutable so return the updated value
        # print(f'-p_iI: {- np.expm1(-tmp1)}',)
        return fun - np.expm1(-tmp1).sum()

    @staticmethod
    @njit(parallel=True)
    def exp_edges(num_edges_i, param, prop_out_I, prop_in_I, prop_out_R, prop_in_R, prop_dyad, selfloops, fit_method):

        N = len(prop_out_I)

        # Preallocate result vectors for each outer loop iteration (i)
        # These arrays store intermediate totals per i, which can be summed later
        f_vector = np.zeros(N)

        # Outer loop is parallelized with prange
        # This is the correct and efficient use of numba's parallelism
        for i in prange(N):
            
            # Use scalar accumulators for better memory efficiency and cache usage
            f_i = 0.0
            
            prop_out_i, prop_in_i = prop_out_I[i], prop_in_I[i]
            
            if "intra" in fit_method:
                # print(f'\n-i: {i}',)
                f_i = num_edges_i(f_i, param, prop_out_i, prop_in_I)
                # print(f'-f_i: {f_i}',)
                f_i = num_edges_i(f_i, param, prop_out_I, prop_in_i)

                f_i /= 2
                # print(f'-f_i: {f_i}',)

            if "bet" in fit_method:
                # print(f'-bet',)
                f_i = num_edges_i(f_i, param, prop_out_i, prop_in_R)
                f_i = num_edges_i(f_i, param, prop_out_R, prop_in_i)

            # Store per-node results
            f_vector[i] = f_i

        # Sum across all nodes to get final result (parallel reduction is fast for large n)
        f_vector = np.sum(f_vector)

        if "intra" in fit_method and not selfloops:
            f_vector = -num_edges_i(-f_vector, param, prop_out_I, prop_in_I)
            
        return f_vector

    def expected_num_edges(self, recompute=False, unsampled_vI = None, num_frozen_edges = 0):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges") or recompute:

            unsampled_vI = np.zeros(self.num_vertices, dtype=np.bool_) if unsampled_vI is None else unsampled_vI

            self._exp_num_edges = self.exp_edges(
                                                self.num_edges_i,
                                                self.param,
                                                self.prop_out_I,
                                                self.prop_in_I,
                                                self.prop_out_R,
                                                self.prop_in_R,
                                                self.prop_dyad,
                                                self.selfloops,
                                                self.fit_method,
            )


            self._exp_num_edges += num_frozen_edges

        return self._exp_num_edges
    
    def sample_wrapper(self, g, gI, unsampled_vI, frozen_edges, measures, recompute = False):
        """
        Define all the variables needed for sampling
        """

        from tqdm import trange
        import os

        # def var for self vars
        mod_vars = self.__dict__
        vsplit, vsplits = gI.vsplit, self.vsplits
        num_graph_samples_per_vsplit, chunk_row_size = self.num_graph_samples_per_vsplit, self.chunk_row_size
        
        # find the starting index graph (0 if a collector routine is not implemented)
        num_start_graph = self.set_ensemble_variables(measures, num_graph_samples_per_vsplit)

        # Note: the seed = vsplit only works for solo-agent. The graph-sampling is parallelized, so it would be random even if vsplit specified.
        
        # set the file name where to store the page-rank and degrees
        # save also the full degrees since they are neede for the ccdf plot
        # measures = [x for x in measures if "degree" not in x]
        fname = self.vars_dir + "/" + "_".join([x.strip("_") for x in measures]) + "_std_on_full_net.pkl"
        if not os.path.exists(fname) or recompute:
            
            print(f'-Sampling vsplit {vsplit} for {measures}: {num_start_graph} graphs already sampled, {np.clip(num_graph_samples_per_vsplit - num_start_graph, 0, None)} remaining')
            for graph_idx in trange(num_start_graph, num_graph_samples_per_vsplit, 
                            desc=f"-Total progress {int(np.round((vsplit+1)/len(vsplits) * 100))}%, Inner Graph Sampling", position = 0, leave= True):

                # sample
                gs = self.sample(ref_g = g, 
                                unsampled_vI = unsampled_vI, 
                                frozen_edges = frozen_edges, 
                                graph_idx = graph_idx, 
                                chunk_row_size = chunk_row_size)
                
                # now calculate the needed
                gs.calculate_measures(g, measures)
                
                # # restrict the pr only on I
                gs.set_pr_on_I(gI)
                
                # save the ensemble average and std for every measures on the self class
                for m in measures:
                    if num_start_graph == graph_idx == 0:
                        mod_vars[f"prev{m}"], mod_vars[f"prev{m}_std"] = 0, 0
                    mod_vars[f"prev{m}"], mod_vars[f"prev{m}_std"] = \
                            self.recursive_mean_std(graph_idx, mod_vars[f"prev{m}"], mod_vars[f"prev{m}_std"], gs.__dict__[m])

            for m in measures:
                mod_vars[m] = mod_vars[f"prev{m}"]
                mod_vars[m+"_std"] = mod_vars[f"prev{m}_std"]

            # save the dict of measures with std
            measures_std = measures.copy()
            measures_std.extend([x + "_std" for x in measures])
            meas_dict = {k:mod_vars[k] for k in mod_vars if k in measures_std}
            utils.save_dict(fname, meas_dict)
        
        else:
            print(f'-Loading {measures} and std for vsplit {vsplit} over {num_graph_samples_per_vsplit } graphs')
            meas_dict = utils.load_dict(fname)
            mod_vars.update(meas_dict)

    def topN_overlap_pr_tot_rel_err_over_mean(self, g, gI):
        
        topN_arr = lambda v: [v[:i] for i in gI._intervals]
        
        # 1. Calculate overlap between g_topN_pr_on_I_rank and model_topN_pr_on_I_rank
        overlap_perc = lambda r: np.array([np.intersect1d(g_topN, exp_topN).size / g_topN.size for g_topN, exp_topN in zip(g._topN_pr_on_I_rank, r)])
        topN_rel_err = lambda r: np.array([utils.tot_rel_err(exp_topN, g_topN) * 100 for g_topN, exp_topN in zip(g._topN_pr, r)])
 
        # obtain the slicing of the page-rank ranking with respect to intervals
        model_topN_pr_on_I_rank = topN_arr(self._pr_on_I_rank)

        # calculate the overlap
        self._topN_overlap_pr_over_mean = overlap_perc(model_topN_pr_on_I_rank)

        # obtain the slicing of the relative error ranking with respect to intervals
        model_topN_pr = topN_arr(self._pr_on_I)
        self._topN_tot_rel_err_over_mean = topN_rel_err(model_topN_pr)


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
            return -np.inf
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
                        return -np.inf
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
            return -np.inf

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
