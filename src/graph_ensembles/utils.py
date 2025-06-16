from .dependencies import *
def n_possible_part(n, n_inner_cl):
	"""
	Number of possible partitions if we start coarse-graining at random n nodes (at level 0) into communities of n_inner_cl nodes each
	"""

	from math import modf
	log_n, log_b = jnp.log(n), jnp.log(n_inner_cl)
	fr_ , int_ = modf(log_n / log_b)
	lo_clust = clust_labels.size - n_inner_cl**int_
	return int_, fr_, lo_clust

def uu_fun(arr):
	# unique values of arr in order of appearance
	return list(map(int, dict.fromkeys(arr)))

def sample_from_p(P, sym = True, seed = None, name = None):

	if not isinstance(P, np.ndarray):
		raise ValueError("Matrix must be an np.ndarray")
	if sym:
		#clever sampling
		P = np.tril(P)
		if seed is not None:
			np.random.seed(seed)
		R = np.random.random_sample(P.shape)
		A = (R < P)
		adj_bin = np.ones(P.shape)
		adj_bin[~A] = 0
		adj_bin = np.tril(adj_bin) + np.triu(adj_bin.T, 1)

		#print(f'-check_symmetric(adj_bin): {check_symmetric(adj_bin)}',)

	#total_links = np.sum(adj_bin) / 2
	#deg = np.sum(adj_bin, axis = 1)

	if name.endswith("Gleditsch"):
		np.fill_diagonal(adj_bin, 1)
	
	return adj_bin#, deg, total_links

def nodes_from(pdf, id_code = None, level = 0):
	"""
	Get the nodes of the pdf
	"""
	if id_code is None:
		id_code = pdf.columns[0].split("_")[-1]
	
	col0_name = f'payer_{id_code}'
	col1_name = f'beneficiary_{id_code}'
	if level > 0:
		col0_name = f'payer_{id_code}_{level}'
		col1_name = f'beneficiary_{id_code}_{level}'

	return np.unique(pdf.loc[:, [col0_name, col1_name]].to_numpy().ravel('K'))

def string_replace(s, replace_dict):
	""" Replace all the replace_dict keys with values.
	Note: to avoid embarassing fails stripe // -> / and the last / if any
	"""
	
	replace_dict["//"] = "/"
	for r in replace_dict:
		s = s.replace(r, replace_dict[r])

	if s.endswith("/"):
		s = s[:-1]
	return s

def set_name_for_plots(name, ref_model):
	'''
		plot the bars and replace the name sum-maxlMSM with maxlMSM since we will plot ONLY sum-models. Thus, put it in the title
		remove "sum-" from all the models as the whole plot will report only Summed models
		if the objective is "Network Reconstruction" also remove the dimension "-1"
	'''
	
	name = name.replace("sum-", "").replace("fine-", "")
	if ref_model.objective == 'NetRec': 
	 	name = name.replace("-1", "")
	if ref_model.objective == "NodeEmb":
		name = name.replace("maxl", "")
  
	return name

def is_number(s):
	""" Check if a string is a number 
	Doing meas.split('.')[-1] when meas = ".95" (confidence = .95) returns 95 and not None
	Hence, no extension will be placed """
	try:
		float(s)  # Try to convert the string to a float
		return True
	except ValueError:
		return False

def prefix_in_(name, prefix_list = None):
	if any([prefix_list == None]):
		prefix_list = ["stripe", "topw"]
	return any([str_ in name for str_ in prefix_list])

def full_path_retriever(ref_model, level = None, name = None, str_dimXBC = None, meas = "pmatrix", ensemble_avg = False, stripes_level = None):
	""" Find the path associated to the arguments one needs """
	
	# load the dictionary of replacements for the level, model name and dimensions
	replace_dict = {}
	if name: 
		replace_dict.update({ref_model.name : name})

	replace_dict.update({f"level{ref_model.level}" : "" if level == None else f"level{level}",})

	if str_dimXBC:
		if ref_model.name.endswith("LPCA"):
			replace_dict.update({f"dimB{ref_model.dimB}/dimC{ref_model.dimC}" : f"{str_dimXBC}"})
		else:
			replace_dict.update({f"dimX{ref_model.dimX}" : f"{str_dimXBC}"})


	# start from loading the directory of the model
	vars_dir = ref_model.vars_dir
	
	# change the stripes_level if needed depending on the name
	if name:
		# if the ref_model has "stripes_level", but not the model_name doesn't need it. Then, remove it
		stripe_dir = ""
		cancel_stripes_level = (not prefix_in_(name)) and prefix_in_(ref_model.name)
		if cancel_stripes_level:
			replace_dict[f"stripes_level{ref_model.stripes_level}"] = ""
		
		# if the name need stripes_level, but the ref_model doesn't have it. Then, add it
		add_stripes_level = prefix_in_(name) and not prefix_in_(ref_model.name)
		if add_stripes_level:
			top_level_dir = f"top_level{ref_model.top_level}"
			replace_dict[f"{top_level_dir}"] = top_level_dir+f"/stripes_level{stripes_level}"
	
	# add the appendix to load the ensemble measurements
	ens_dir = ""
	if ensemble_avg:
		ens_dir = "/ensemble"
		meas = "ens_avg_lbci_ubci_" + meas
	
	# now replace the model dir parts and add ens_dir
	replace_path = string_replace(vars_dir,replace_dict) + ens_dir

	# check if meas has an extension ("X.pt"). If not, add ".csv" since it is the most common format
	if meas:
		extension = meas.split('.')[-1] if '.' in meas else None
		if not extension or is_number(extension):
			replace_path += f"/{meas}.csv"
		else:
			replace_path += f"/{meas}"

	
	
	return replace_path

# Plot Binary Measures
def save_fig(fig, full_path = None, save = True):
	if save:
		import os
		dir_ = os.path.dirname(full_path)
		os.makedirs(dir_, exist_ok = True)
		#print(f'-dir: {dir}',)
		
		fig.savefig(full_path, dpi = 100, facecolor = "white", bbox_inches = "tight")

def get_reduced_by(folder_path):
	"""Find the reduced_by of another model found via full_path_retriever"""
	import os

	# defined the filter suffix
	suffix = "deg_annd_cc.pkl"

	# List all entries in the given folder path
	entries = os.listdir(folder_path)

	# remove the suffix from the entries
	reduced_by = [entry[:len(entry)-len(suffix)] for entry in entries if entry.endswith(suffix)]

	return reduced_by[0]


# Directed Utils

def check_dim_max_auc(kwargs, model_kwargs):
	from Directed_Graph import Directed_Graph
	check_kwargs = kwargs.copy()
	check_kwargs.update(model_kwargs)
	fake4dir = Directed_Graph(**check_kwargs)
	
	from pathlib import Path
	par_dir = Path(fake4dir.vars_dir).parents[1]
	dim_pr = par_dir / "dim_max_test_auc_pr.csv"
	dim_roc = par_dir / "dim_max_test_auc_roc.csv"
	
	if dim_pr.exists() or dim_roc.exists():
		# upload the results
		dim_pr = np.genfromtxt(dim_pr, delimiter=',')
		dim_roc = np.genfromtxt(dim_roc, delimiter=',')
		
		if dim_pr == dim_roc:
			return False, dim_pr
		else:
			print('-dim_pr \neq dim_roc', dim_pr, dim_roc)
			return False, dim_pr

def to_tensor(X):
	""" Convert numpy array to tensor """
	if not tc.is_tensor(X):
		return tc.from_numpy(X)
	return X

def signed_rel_err(x, y):
	""" Relative error among each element of x and y. It returns an array
	"""
	return (x-y) / y


def rel_err(x, y):
	""" Relative error between x and y. It returns a scalar """
	return np.linalg.norm(x - y) / np.linalg.norm(y)

def fc_title(ref_model):
	return "Summed" if ref_model.fc_direction.startswith("fc") else "Fractioned"

def str_dimXBC_(name, dim): 
	return f"dimB{dim[0]}/dimC{dim[1]}" if name.endswith("LPCA") else f"dimX{dim}"

def load_array(full_path):
	""" Load an array """
	if full_path.endswith("csv"):
		return np.genfromtxt(full_path, delimiter = ",")

def load_meas(ref_model, level, name, str_dimXBC = None, meas = "pmatrix", ensemble_avg = False):
	full_path = full_path_retriever(ref_model, level, name, str_dimXBC, meas, ensemble_avg)

	# use different opener for different formats
	if full_path.endswith("csv"):
		return np.genfromtxt(full_path, delimiter = ",")
	elif full_path.endswith("pkl"):
		return pd.read_pickle(full_path)
	elif full_path.endswith("txt"):
		return open(full_path, "r").read()

# multiprocess trial functions
def worker(args):
    import multiprocessing
    start, end, N, p_ij, param, prop_out, prop_in, prop_dyad, selfloops = args
    rows = []
    cols = []
    # logs = []
    worker_id = multiprocessing.current_process().name
    # logs.append(f'Worker {worker_id} started: processing indices {start} to {end}')
    for flat_idx in range(start, end):
        i = flat_idx // N
        j = flat_idx % N
        if not selfloops and i == j:
            continue
        p = p_ij(param, prop_out[i], prop_in[j], prop_dyad(i, j))
        if np.random.random() < p:
            rows.append(i)
            cols.append(j)
    # logs.append(f'Worker {worker_id} finished.')
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)#, logs

def parallel_sample(p_ij, param, prop_out, prop_in, prop_dyad, selfloops, num_procs=3):
	"""Sample edges in parallel using multiple processes."""
	from multiprocessing import Pool
	
	N = len(prop_out)
	total_ops = N * N  # or N * (N - 1) if not selfloops
	num_chunks = total_ops // num_procs
	tasks = []
	for t in range(num_procs):
		start = t * num_chunks
		end = (t + 1) * num_chunks if t < num_procs - 1 else total_ops
		tasks.append((start, end, N, p_ij, param, prop_out, prop_in, prop_dyad, selfloops))
		
	with Pool(processes=num_procs) as pool:
		results = pool.map(worker, tasks)
	rows = [r for r, _ in results]
	cols = [c for _, c in results]
	# logs = [log for _, _, log in results]
	# for log in logs:
	# 	for line in log:
	# 		print(line)
	
	return np.concatenate(rows), np.concatenate(cols)