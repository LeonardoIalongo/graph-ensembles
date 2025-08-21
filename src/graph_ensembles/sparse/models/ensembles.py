""" This module defines the classes that allow for the construction of
network ensembles from partial information. They can be used for
reconstruction, filtering or pattern detection among others. """
from .. import graphs
import numpy as np
import numpy.random as rng
import scipy.sparse as sp
from numba import jit, njit, prange, get_num_threads, get_thread_id
from math import isinf
from numba import float64
from numba.experimental import jitclass
from numba.typed import List
import torch as tc
from ... import utils


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


class GraphEnsemble():
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
    reference graph but not all (e.g. only the num_edges, but keeping strengths
    the same).

    """

    @staticmethod
    @njit()  # pragma: no cover
    def prop_dyad(i, j):
        """Define empy dyadic property as it is not always defined."""
        return 1.0
    
    def get(self, var_name):
        """Return the variable if it exists, otherwise return None.
        Note that for inner variables one should get the _var_name. That is why we check also _name."""
        
        return self.__dict__.get(var_name)

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
        
    def expected_num_edges(self, recompute=False, unsampled_vI = None):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges") or recompute:

            unsampled_vI = np.zeros(self.num_vertices, dtype=np.bool_) if unsampled_vI is None else unsampled_vI

            self._exp_num_edges = self.exp_edges(
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.prop_dyad,
                self.selfloops,
                unsampled_vI
            )

        return self._exp_num_edges

    def expected_degree(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_degree") or recompute:
            res = self.exp_degrees(
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.prop_dyad,
                self.selfloops,
            )
            self._degree = res[0]
            self._out_degree = res[1]
            self._in_degree = res[2]

        return self._degree

    def expected_out_degree(self, recompute=False):
        """Compute the expected out degree."""
        if not hasattr(self, "_out_degree") or recompute:
            _ = self.expected_degree(recompute=recompute)

        return self._out_degree

    def expected_in_degree(self, recompute=False):
        """Compute the expected in degree."""
        if not hasattr(self, "_in_degree") or recompute:
            _ = self.expected_degree(recompute=recompute)

        return self._in_degree

    def expected_av_nn_property(
        self, prop, ndir="out", selfloops=False, deg_recompute=False
    ):
        """Computes the expected value of the nearest neighbour average of
        the property array. The array must have the first dimension
        corresponding to the vertex index.
        """
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

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

        # Compute expected property
        av_nn = self.exp_av_nn_prop(
            self.p_ij,
            self.param,
            self.prop_out,
            self.prop_in,
            self.prop_dyad,
            prop,
            ndir,
            self.selfloops,
        )

        # Test that mask is the same
        ind = deg != 0
        msg = "Got a av_nn for an empty neighbourhood."
        assert np.all(av_nn[~ind] == 0), msg

        # Average results
        av_nn[ind] = av_nn[ind] / deg[ind]

        # Restore model self-loops properties if they have been modified
        if tmp_self != self.selfloops:
            self.selfloops = tmp_self
            if hasattr(self, "_out_degree"):
                del self._out_degree
            if hasattr(self, "_in_degree"):
                del self._in_degree
            if hasattr(self, "_degree"):
                del self._degree

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
            raise Exception("Model must be fitted beforehand.")

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
                if hasattr(self, "_out_degree"):
                    del self._out_degree
                if hasattr(self, "_in_degree"):
                    del self._in_degree
                if hasattr(self, "_degree"):
                    del self._degree

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

        return like

    def send_variables_to_gpu(self, arr):
        """Send the arrays to one GPU for efficient calculations"""
        dtype, device = tc.float32, "cuda:0"
        self.param = tc.from_numpy(self.param).to(device, dtype = dtype)
        self.prop_out = tc.from_numpy(self.prop_out).to(device, dtype = dtype)
        self.prop_in = tc.from_numpy(self.prop_in).to(device, dtype = dtype)
        self.selfloops = tc.tensor(self.selfloops).to(device)
        return tc.from_numpy(arr).to(device)

    def sample(
        self,
        ref_g=None,
        weights=None,
        out_strength=None,
        in_strength=None,
        selfloops=None,
        unsampled_vI=None,
        frozen_edges=None,
        graph_idx=0,
        chunk_row_size=1000,
    ):
        """Return a Graph sampled from the ensemble.

        If a reference graph is passed (ref_g) then the properties of the graph
        will be copied to the new samples.
        """
        from os import makedirs
        
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

        # define the g vars_dir
        g.vars_dir = self.vars_dir_ensembles + f"/samples/graph{graph_idx}"
        
        # Check if reference graph is available
        if ref_g is not None:
            if hasattr(ref_g, "num_groups"):
                g.num_groups = ref_g.num_groups
                g.group_dict = ref_g.group_dict
                g.group_dtype = ref_g.group_dtype
                g.groups = ref_g.groups

        else:
            g.id_dict = {}
            for i in range(g.num_vertices):
                g.id_dict[i] = i

        # Sample edges
        g.weighted = None
        if weights is None:
            unsampled_vI = np.zeros(self.num_vertices, dtype=np.bool_) if unsampled_vI is None else unsampled_vI
            rows, cols = self._binary_sample(
                self.tc_p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.tc_prop_dyad,
                self.selfloops,
                unsampled_vI,
                frozen_edges,
                chunk_row_size,
                seed=graph_idx,
            )
            vals = np.ones(len(rows), dtype=bool)
        elif weights == "cremb":
            g.weighted = True
            if out_strength is None:
                out_strength = self.prop_out
            if in_strength is None:
                in_strength = self.prop_in
            rows, cols, vals = self._cremb_sample(
                self.p_ij,
                self.param,
                self.prop_out,
                self.prop_in,
                self.prop_dyad,
                out_strength,
                in_strength,
                self.selfloops,
            )
        else:
            raise ValueError("Weights method not recognised or implemented.")

        makedirs(g.vars_dir, exist_ok = True) 

        # Convert to adjacency matrix
        g.adj = sp.csr_array(
            (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
        )

        # calculate num_edges and degrees
        # g.num_edges()
        # g.out_degree()

        if ref_g != None:
            g.ivec = g.pagerank_power(**ref_g._kwargs_pr)

        g.save_vars(name = f"graph{graph_idx}")

        return g

    @njit(parallel=True)
    def block_parallel_sample(p_ij, param, prop_out, prop_in, prop_dyad, selfloops, unsampled_vI):
        """
        Samples the internal index (not the nodes - id) according to the probability pij.
        Avoid all the pairs involving unsampled_vI, since they will be added afterwards
        """
        N = len(prop_out)
        total_ops = N * N
        num_blocks = get_num_threads()
        block_size = total_ops // num_blocks

        # Preallocate buffers for each block/thread
        thread_rows = List()
        thread_cols = List()

        num_elements_unsampled_vI = np.sum(unsampled_vI)

        for _ in range(num_blocks):
            thread_rows.append(List.empty_list(np.int64))
            thread_cols.append(List.empty_list(np.int64))

        # np.random.seed(1)
        for block in prange(num_blocks):
            start = block * block_size
            end = (block + 1) * block_size if block < num_blocks - 1 else total_ops
            thread_id = get_thread_id()
            for flat_idx in range(start, end):
                i = flat_idx // N
                j = flat_idx % N

                # if True, skipping the (i,j) sampling
                # 1) selfloops; 2) freezed connections
                if (not selfloops and i == j):
                    continue
                if num_elements_unsampled_vI > 0:
                    if unsampled_vI[i] and unsampled_vI[j]:
                        continue
                p = p_ij(param, prop_out[i], prop_in[j], prop_dyad(i, j))
                if np.random.random() < p:
                    thread_rows[thread_id].append(i)
                    thread_cols[thread_id].append(j)

        return thread_rows, thread_cols

    def tc_p_ij(self, d, x_i, y_j, z_ij = 1):
        """Compute the probability of connection between node i and j."""
        tmp = d * x_i * y_j
        return -tc.expm1(-tmp)

    def tc_prop_dyad(self, i, j):
        return 1  # dummy dyad

    def tc_block_parallel_sample_improved(self, p_ij, param, prop_out, prop_in, prop_dyad, selfloops, unsampled_vI, chunk_row_size=1000, seed=0):
        """
        Efficiently sample edges in chunks using broadcasting to avoid memory overflow.
        """
        N = prop_out.shape[0]
        device = prop_out.device

        all_rows = []
        all_cols = []

        # Seed once at the beginning
        if device.type == 'cuda':
            tc.cuda.manual_seed(seed)
        else:
            tc.manual_seed(seed)

        # Process in chunks to avoid memory issues
        for start in range(0, N, chunk_row_size):
            end = min(start + chunk_row_size, N)
            # Slicing is cheap and doesn't copy data immediately
            x_i_chunk = prop_out[start:end]
            current_chunk_size = x_i_chunk.shape[0]

            # 1. Use broadcasting to compute probabilities for the entire block
            # x_i_chunk.view(-1, 1) has shape [current_chunk_size, 1]
            # prop_in.view(1, -1) has shape [1, N]
            # The result `p_matrix` will have shape [current_chunk_size, N]
            
            # Note: prop_dyad needs to be handled. Since it's a dummy 1, it's easy.
            # If it were complex, it would need to operate on broadcasted inputs.
            z_ij = prop_dyad(None, None) # Dummy call, gets 1

            # Calculate probability matrix directly
            p_matrix = p_ij(
                param,
                x_i_chunk.view(-1, 1), # Shape: [chunk_size, 1]
                prop_in.view(1, -1),   # Shape: [1, N]
                # z_ij
            )
            
            # 2. Apply masks directly to the probability matrix
            
            # Mask for self-loops
            if not selfloops:
                # Create indices for the diagonal within this chunk
                # The columns to zero-out are from `start` to `end-1`
                diag_rows = tc.arange(current_chunk_size, device=device)
                diag_cols = tc.arange(start, end, device=device)
                p_matrix[diag_rows, diag_cols] = 0.0

            # Mask for unsampled vertices
            if unsampled_vI.any():
                # Get the unsampled status for the current chunk of rows
                u_i_chunk = unsampled_vI[start:end] # Shape: [chunk_size]
                
                # Find rows in the chunk that are "unsampled"
                unsampled_rows_mask = u_i_chunk.view(-1, 1) # Shape: [chunk_size, 1]
                # Find columns in the full graph that are "unsampled"
                unsampled_cols_mask = unsampled_vI.view(1, -1) # Shape: [1, N]
                
                # Create a boolean matrix where True means both i and j are unsampled
                # This uses broadcasting again
                full_mask = unsampled_rows_mask & unsampled_cols_mask
                
                # Set probability to 0 where the mask is True
                p_matrix[full_mask] = 0.0
            
            # 3. Sample from the probability matrix
            # Generate random numbers for the whole matrix at once
            rand_matrix = tc.rand_like(p_matrix)
            
            # Find where sampling was successful
            # `tc.where` returns a tuple of tensors (rows, cols)
            sampled_rows_in_chunk, sampled_cols = tc.where(rand_matrix < p_matrix)
            
            # If any edges were sampled in this chunk
            if sampled_rows_in_chunk.shape[0] > 0:
                # Convert row indices from chunk-local (0 to chunk_size-1) to global
                sampled_rows_global = sampled_rows_in_chunk + start
                
                # Append tensors directly on the GPU
                all_rows.append(sampled_rows_global)
                all_cols.append(sampled_cols)

        # 4. Concatenate all results on the GPU, then transfer to CPU once
        if all_rows:
            rows_gpu = tc.cat(all_rows)
            cols_gpu = tc.cat(all_cols)
            rows = rows_gpu.cpu().numpy()
            cols = cols_gpu.cpu().numpy()
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
            
        return rows, cols
    
    def tc_block_parallel_sample(self, p_ij, param, prop_out, prop_in, prop_dyad, selfloops, unsampled_vI, chunk_row_size=100, seed = 0):
        """
        Efficiently sample edges in chunks to avoid GPU memory overflow.
        """
        import torch as tc

        N = prop_out.shape[0]
        device = prop_out.device

        # Seed once at the beginning
        if device.type == 'cuda':
            tc.cuda.manual_seed(seed)
        else:
            tc.manual_seed(seed)

        # Prepare output lists to collect tensors directly on GPU
        all_rows_gpu = []
        all_cols_gpu = []

        # Process in chunks to avoid memory issues
        for start in range(0, N, chunk_row_size):
            end = min(start + chunk_row_size, N)
            idx_i = tc.arange(start, end, device=device)
            idx_j = tc.arange(N, device=device)


            # get the flat list of all the pairs
            grid_i, grid_j = tc.meshgrid(idx_i, idx_j, indexing='ij')
            grid_i = grid_i.flatten()
            grid_j = grid_j.flatten()

            # Mask some pairs
            mask = tc.ones_like(grid_i, dtype=tc.bool)

            # if False, when grid_i == grid_j --> mask = False
            if not selfloops:
                mask &= (grid_i != grid_j)
            # Set mask = False when both unsampled_vI are True
            if unsampled_vI.any():
                mask &= ~(unsampled_vI[grid_i] & unsampled_vI[grid_j])

            # Retain the pairs to sample
            grid_i = grid_i[mask]
            grid_j = grid_j[mask]

            # Compute probabilities over the unmasked pairs
            x_i, y_j = prop_out[grid_i], prop_in[grid_j]
            p = p_ij(param, x_i, y_j, prop_dyad(grid_i, grid_j))

            # Sample edges
            rand = tc.rand_like(p)
            selected_idx = tc.where(rand < p)
            grid_i = grid_i[selected_idx]
            grid_j = grid_j[selected_idx]

            # Append tensors directly on the GPU
            all_rows_gpu.append(grid_i)
            all_cols_gpu.append(grid_j)

        # Concatenate all results on the GPU, then transfer to CPU once
        if all_rows_gpu: # Check if list is not empty
            rows_gpu = tc.cat(all_rows_gpu)
            cols_gpu = tc.cat(all_cols_gpu)

            rows = rows_gpu.cpu().numpy()
            cols = cols_gpu.cpu().numpy()
        else:
            # No edges sampled, return empty arrays
            rows = tc.empty(0, dtype=tc.int64).numpy()
            cols = tc.empty(0, dtype=tc.int64).numpy()


        return rows, cols

    def _binary_sample(self, p_ij, param, prop_out, prop_in, prop_dyad, selfloops, unsampled_vI, frozen_edges, chunk_row_size, seed):
        
        from itertools import chain
        import pandas as pd

        rows, cols = self.tc_block_parallel_sample_improved(
                                                p_ij, 
                                                param, 
                                                prop_out, 
                                                prop_in,
                                                prop_dyad, 
                                                selfloops, 
                                                unsampled_vI,
                                                chunk_row_size,
                                                seed
                                            )

        # Flatten rows/cols (List[List[int64]]) to 1D numpy arrays
        # rows = np.fromiter(chain.from_iterable(sampled_rows), dtype=np.int64)
        # cols = np.fromiter(chain.from_iterable(sampled_cols), dtype=np.int64)

        assert isinstance(frozen_edges, (pd.DataFrame, type(None))), "frozen_edges must be a pandas.DataFrame or None"
        # add frozen edges into the sampled matrix
        if isinstance(frozen_edges, pd.DataFrame):
            # map nodes ids to internal idx of src_vI, dst_vI
            src_vI, dst_vI = frozen_edges.T.values

            # concatenate
            rows = np.concatenate((rows, src_vI))
            cols = np.concatenate((cols, dst_vI))
        
        return rows, cols
    
    @staticmethod
    @njit(parallel=True)
    def exp_edges(p_ij, param, prop_out, prop_in, prop_dyad, selfloops, unsampled_vI):
        """Compute the objective function of the num_edges solver and its
        derivative.
        """
        N = len(prop_out)

        # Preallocate result vectors for each outer loop iteration (i)
        # These arrays store intermediate totals per i, which can be summed later
        f_vector = np.zeros(N)

        # Outer loop is parallelized with prange
        # This is the correct and efficient use of numba's parallelism
        for i in prange(N):
            p_out_i = prop_out[i]

            # Use scalar accumulators for better memory efficiency and cache usage
            f_i = 0.0
            
            # Inner loop is serial — this is good because nested prange is not well supported
            for j in range(N):

                if unsampled_vI[i] and unsampled_vI[j]:
                    continue
                
                if (i != j) or selfloops:
                    p_in_j = prop_in[j]

                    # Call the user-defined function to get value and jacobian for dyad (i, j)
                    p_val = p_ij(param, p_out_i, p_in_j, prop_dyad(i,j))

                    # Accumulate results efficiently in scalars
                    f_i += p_val

            # Store per-node results
            f_vector[i] = f_i

        # Sum across all nodes to get final result (parallel reduction is fast for large n)
        f_vector = np.sum(f_vector)
        
        return f_vector
    
    @staticmethod
    @njit(parallel=True)  # pragma: no cover
    def exp_degrees(p_ij, param, prop_out, prop_in, prop_dyad, selfloops):
        """Compute the expected total, out-degree and in-degree sequences."""
        num_v = len(prop_out)
        exp_d = np.zeros(num_v, dtype=np.float64)
        exp_d_out = np.zeros(num_v, dtype=np.float64)
        exp_d_in = np.zeros(num_v, dtype=np.float64)

        for i in prange(num_v):
            p_out_i = prop_out[i]
            for j in range(num_v):
                p_in_j = prop_in[j]
                if i != j:
                    pij = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    exp_d_out[i] += pij
                    
                    # undirected degrees
                    pji = p_ij(param, prop_out[j], prop_in[i], prop_dyad(i, j))
                    qij = pij + pji - pij * pji
                    exp_d[i] += qij
                    exp_d[j] += qij
                    
                elif selfloops:
                    pii = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    exp_d_out[i] += pii
        
        for l in prange(num_v):
            p_in_l = prop_in[l]
            for m in range(num_v):
                p_out_m = prop_out[m]
                if l != m:
                    pml = p_ij(param, p_out_m, p_in_l, prop_dyad(m, l))
                    exp_d_in[l] += pml
                    
                    # undirected degrees
                    # pji = p_ij(param, prop_out[j], prop_in[i], prop_dyad(i, j))
                    # qij = pij + pji - pij * pji
                    # exp_d[i] += qij
                    # exp_d[j] += qij
                    
                elif selfloops:
                    pll = p_ij(param, p_in_l, p_in_l, prop_dyad(l, l))
                    exp_d_in[l] += pll
                    
        return exp_d, exp_d_out, exp_d_in


    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_av_nn_prop(
        p_ij, param, prop_out, prop_in, prop_dyad, prop, ndir, selfloops
    ):
        """Compute the expected average nearest neighbour property."""
        av_nn = np.zeros(prop.shape, dtype=np.float64)
        for i, p_out_i in enumerate(prop_out):
            p_in_i = prop_in[i]
            for j in range(i):
                p_out_j = prop_out[j]
                p_in_j = prop_in[j]
                pij = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                pji = p_ij(param, p_out_j, p_in_i, prop_dyad(j, i))
                if ndir == "out":
                    av_nn[i] += pij * prop[j]
                    av_nn[j] += pji * prop[i]
                elif ndir == "in":
                    av_nn[i] += pji * prop[j]
                    av_nn[j] += pij * prop[i]
                elif ndir == "out-in":
                    p = 1 - (1 - pij) * (1 - pji)
                    av_nn[i] += p * prop[j]
                    av_nn[j] += p * prop[i]
                else:
                    raise ValueError("Direction of neighbourhood not right.")

        if selfloops:
            for i in range(len(prop_out)):
                pii = p_ij(param, prop_out[i], prop_in[i], prop_dyad(i, j))
                if ndir == "out":
                    av_nn[i] += pii * prop[i]
                elif ndir == "in":
                    av_nn[i] += pii * prop[i]
                elif ndir == "out-in":
                    av_nn[i] += pii * prop[i]
                else:
                    raise ValueError("Direction of neighbourhood not right.")

        return av_nn

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _likelihood(
        logp, log1mp, param, prop_out, prop_in, prop_dyad, adj_i, adj_j, selfloops
    ):
        """Compute the binary log likelihood of a graph given the fitted model."""
        like = 0
        for i, p_out_i in enumerate(prop_out):
            n = adj_i[i]
            m = adj_i[i + 1]
            j_list = adj_j[n:m]
            for j, p_in_j in enumerate(prop_in):
                if (i != j) | selfloops:
                    # Check if link exists
                    if j in j_list:
                        tmp = logp(param, p_out_i, p_in_j, prop_dyad(i, j))
                    else:
                        tmp = log1mp(param, p_out_i, p_in_j, prop_dyad(i, j))

                    if isinf(tmp):
                        return tmp
                    like += tmp

        return like

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample(
        p_ij, param, prop_out, prop_in, prop_dyad, s_out, s_in, selfloops
    ):
        """Sample from the ensemble with weights from the CremB model."""
        s_tot = np.sum(s_out)
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in) / s_tot) < 1e-6, msg
        rows = []
        cols = []
        vals = []
        for i, p_out_i in enumerate(prop_out):
            for j, p_in_j in enumerate(prop_in):
                if (i != j) | selfloops:
                    p = p_ij(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        vals.append(rng.exponential(s_out[i] * s_in[j] / (s_tot * p)))

        return rows, cols, vals


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

    def expected_num_edges_label(self, recompute=False):
        """Compute the expected number of edges."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_exp_num_edges_label") or recompute:
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

            # Compute measure
            self._exp_num_edges_label = self.exp_edges_label(
                self.p_ijk,
                self.exp_edges_layer,
                self.param,
                prop_out,
                prop_in,
                self.prop_dyad,
                self.selfloops,
            )

        return self._exp_num_edges_label

    def expected_degree_by_label(self, recompute=False):
        """Compute the expected undirected degree."""
        if not hasattr(self, "param"):
            raise Exception("Model must be fitted beforehand.")

        if not hasattr(self, "_degree_label") or recompute:
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

            # Compute measure
            res = self.exp_degrees_label(
                self.p_ijk,
                self.param,
                prop_out,
                prop_in,
                self.prop_dyad,
                self.num_vertices,
                self.num_labels,
                self.selfloops,
            )
            self._degree_label = sp.dok_array((self.num_vertices, self.num_labels))
            self._degree_label._update(res[0])
            self._degree_label = self._degree_label.tocsr()
            self._out_degree_label = sp.dok_array(
                (self.num_vertices, self.num_labels)
            )
            self._out_degree_label._update(res[1])
            self._out_degree_label = self._out_degree_label.tocsr()
            self._in_degree_label = sp.dok_array(
                (self.num_vertices, self.num_labels)
            )
            self._in_degree_label._update(res[2])
            self._in_degree_label = self._in_degree_label.tocsr()

        return self._degree_label

    def expected_out_degree_by_label(self, recompute=False):
        """Compute the expected out degree."""
        if not hasattr(self, "_out_degree_label") or recompute:
            _ = self.expected_degree_by_label(recompute=recompute)

        return self._out_degree_label

    def expected_in_degree_by_label(self, recompute=False):
        """Compute the expected in degree."""
        if not hasattr(self, "_in_degree_label") or recompute:
            _ = self.expected_degree_by_label(recompute=recompute)

        return self._in_degree_label

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
                adj = [sp.csr_array(g[i, :, :]) for i in range(g.shape[0])]
                tensor = True
            elif g.ndim == 2:
                adj = sp.csr_array(g)
                tensor = False
            else:
                raise ValueError("Adjacency array is neither 2 nor 3 dimensional.")
        elif isinstance(g, list):
            adj = [sp.csr_array(x) for x in g]
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

            like = np.float64(0.0)
            for i in range(self.num_labels):
                tmp = self._likelihood_layer(
                    self.logp_ijk,
                    self.log1mp_ijk,
                    self.param[i],
                    prop_out[i],
                    prop_in[i],
                    self.prop_dyad,
                    adj[i].indptr,
                    adj[i].indices,
                    selfloops,
                )
                if isinf(tmp):
                    return tmp
                like += tmp
        else:
            like = self._likelihood(
                self.logp,
                self.log1mp,
                self.param,
                self.prop_out,
                self.prop_in,
                self.prop_dyad,
                adj.indptr,
                adj.indices,
                selfloops,
            )

        return like

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
        prop_in = self.csx_to_tuple_list(prop_in.indptr, prop_in.indices, prop_in.data)

        # If weights are given check that they make sense
        if out_strength_label is None:
            s_out_l = prop_out
        else:
            # Check dimensions
            if isinstance(out_strength_label, np.ndarray):
                out_strength_label = sp.csc_array(out_strength_label)

            if sp.issparse(out_strength_label):
                msg = (
                    "Out strength by label must have shape (num_vertices, "
                    "num_labels)."
                )
                assert out_strength_label.shape == (
                    self.num_vertices,
                    self.num_labels,
                ), msg
                tmp = out_strength_label.tocsc()
                s_out_l = self.csx_to_tuple_list(tmp.indptr, tmp.indices, tmp.data)
            else:
                raise ValueError("Out strength by label must be an array.")

        if in_strength_label is None:
            s_in_l = prop_in
        else:
            if isinstance(in_strength_label, np.ndarray):
                in_strength_label = sp.csc_array(in_strength_label)

            if sp.issparse(in_strength_label):
                msg = (
                    "In strength by label must have shape (num_vertices, "
                    "num_labels)."
                )
                assert in_strength_label.shape == (
                    self.num_vertices,
                    self.num_labels,
                ), msg
                tmp = in_strength_label.tocsc()
                s_in_l = self.csx_to_tuple_list(tmp.indptr, tmp.indices, tmp.data)
            else:
                raise ValueError("In strength by label must be an array.")

        # Sample edges by layer
        g.adj = []
        for i in range(self.num_labels):
            if weights is None:
                rows, cols = self._binary_sample_layer(
                    self.p_ijk,
                    self.param[i],
                    prop_out[i],
                    prop_in[i],
                    self.prop_dyad,
                    self.selfloops,
                )
                vals = np.ones(len(rows), dtype=bool)
            elif weights == "cremb":
                rows, cols, vals = self._cremb_sample_layer(
                    self.p_ijk,
                    self.param[i],
                    prop_out[i],
                    prop_in[i],
                    self.prop_dyad,
                    s_out_l[i],
                    s_in_l[i],
                    self.selfloops,
                )
            else:
                raise ValueError("Weights method not recognised or implemented.")

            # Convert to adjacency matrix
            g.adj.append(
                sp.csr_array(
                    (vals, (rows, cols)), shape=(g.num_vertices, g.num_vertices)
                )
            )

        return g

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_label(
        p_ijk, exp_edges_layer, param, prop_out, prop_in, prop_dyad, selfloops
    ):
        """Compute the expected number of edges with one parameter controlling
        for the num_edges for each label.
        """
        num_labels = len(prop_out)
        exp_edges = np.zeros(num_labels, dtype=np.float64)

        for i in range(num_labels):
            exp_edges[i] = exp_edges_layer(
                p_ijk, param[i], prop_out[i], prop_in[i], prop_dyad, selfloops
            )

        return exp_edges

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_edges_layer(p_ijk, param, prop_out, prop_in, prop_dyad, selfloops):
        """Compute the expected number of edges with one parameter controlling
        for the num_edges for each label.
        """
        exp_e = 0.0
        for i, p_out_i in zip(prop_out[0], prop_out[1]):
            for j, p_in_j in zip(prop_in[0], prop_in[1]):
                if (i != j) | selfloops:
                    exp_e += p_ijk(param, p_out_i, p_in_j, prop_dyad(i, j))

        return exp_e

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def exp_degrees_label(
        p_ijk, param, prop_out, prop_in, prop_dyad, num_v, num_l, selfloops
    ):
        """Compute the expected undirected, in and out degree sequences."""
        exp_d = {}
        exp_d_out = {}
        exp_d_in = {}

        for lbl, (p_out, p_in) in enumerate(zip(prop_out, prop_in)):
            for i, p_out_i in zip(p_out[0], p_out[1]):
                loc_i = p_in[0] == i
                if np.any(loc_i):
                    p_in_i = p_in[1][loc_i][0]
                else:
                    p_in_i = 0.0

                for j, p_in_j in zip(p_in[0], p_in[1]):
                    loc_j = p_out[0] == j
                    if np.any(loc_j):
                        p_out_j = p_out[1][loc_j][0]
                    else:
                        p_out_j = 0.0

                    if i != j:
                        pij = p_ijk(param[lbl], p_out_i, p_in_j, prop_dyad(i, j))
                        pji = p_ijk(param[lbl], p_out_j, p_in_i, prop_dyad(j, i))
                        p = pij + pji - pij * pji
                        key = (np.int64(i), np.int64(lbl))
                        if key in exp_d:
                            exp_d[key] += p
                            exp_d_out[key] += pij
                            exp_d_in[key] += pji
                        else:
                            exp_d[key] = p
                            exp_d_out[key] = pij
                            exp_d_in[key] = pji

                        key = (np.int64(j), np.int64(lbl))
                        if key in exp_d:
                            exp_d[key] += p
                            exp_d_out[key] += pji
                            exp_d_in[key] += pij
                        else:
                            exp_d[key] = p
                            exp_d_out[key] = pji
                            exp_d_in[key] = pij

                    elif selfloops:
                        pii = p_ijk(param[lbl], p_out_i, p_in_j, prop_dyad(i, j))
                        key = (np.int64(i), np.int64(lbl))
                        if key in exp_d:
                            exp_d[key] += pii
                            exp_d_out[key] += pii
                            exp_d_in[key] += pii
                        else:
                            exp_d[key] = pii
                            exp_d_out[key] = pii
                            exp_d_in[key] = pii

        return exp_d, exp_d_out, exp_d_in

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _likelihood_layer(
        logp_ijk,
        log1mp_ijk,
        param,
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
                ind = prop_out[0] == i
                if not np.any(ind):
                    return -np.inf
                if np.any(prop_out[1][ind] == 0):
                    return -np.inf

                j_list = indices[n:m]
                for j in j_list:
                    ind = prop_in[0] == j
                    if not np.any(ind):
                        return -np.inf
                    if np.any(prop_in[1][ind] == 0):
                        return -np.inf

        # Now compute likelihood due to non-zero values of pijk
        for i, out_i in zip(prop_out[0], prop_out[1]):
            n = indptr[i]
            m = indptr[i + 1]
            j_list = indices[n:m]
            for j, in_j in zip(prop_in[0], prop_in[1]):
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
    def _binary_sample_layer(p_ijk, param, prop_out, prop_in, prop_dyad, selfloops):
        """Sample from the ensemble."""
        rows = List()
        cols = List()

        for i, p_out_i in zip(prop_out[0], prop_out[1]):
            for j, p_in_j in zip(prop_in[0], prop_in[1]):
                if (i != j) | selfloops:
                    p = p_ijk(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)

        return rows, cols

    @staticmethod
    @jit(nopython=True)  # pragma: no cover
    def _cremb_sample_layer(
        p_ijk, param, prop_out, prop_in, prop_dyad, s_out, s_in, selfloops
    ):
        """Sample from the ensemble with weights from the CremB model."""
        s_tot = np.sum(s_out[1])
        msg = "Sum of in/out strengths not the same."
        assert np.abs(1 - np.sum(s_in[1]) / s_tot) < 1e-6, msg

        rows = List()
        cols = List()
        vals = List()

        for i, p_out_i in zip(prop_out[0], prop_out[1]):
            for j, p_in_j in zip(prop_in[0], prop_in[1]):
                if (i != j) | selfloops:
                    p = p_ijk(param, p_out_i, p_in_j, prop_dyad(i, j))
                    if rng.random() < p:
                        rows.append(i)
                        cols.append(j)
                        ind_out = s_out[0] == i
                        ind_in = s_in[0] == j
                        if np.any(ind_out) or np.any(ind_in):
                            vals.append(
                                rng.exponential(
                                    s_out[1][ind_out][0]
                                    * s_in[1][ind_in][0]
                                    / (s_tot * p)
                                )
                            )
                        else:
                            vals.append(0.0)

        return rows, cols, vals