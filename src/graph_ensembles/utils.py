from .dependencies import *
import numpy as np
import pandas as pd
import graph_ensembles as ge

def n_possible_part(n, n_inner_cl, clust_labels):
    """
    Number of possible partitions if we start coarse-graining at random n nodes (at level 0) into communities of n_inner_cl nodes each
    """
    from math import modf
    log_n, log_b = np.log(n), np.log(n_inner_cl)
    fr_ , int_ = modf(log_n / log_b)
    lo_clust = clust_labels.size - n_inner_cl**int_
    return int_, fr_, lo_clust

def uu_fun(arr):
    # unique values of arr in order of appearance
    return list(map(int, dict.fromkeys(arr)))

def _set_mpl_params(font = "DejaVu Serif", fontsize = '23'):
    """
    Set latex if needed and fontsize
    """
    family = "serif" if font in ["DejaVu Serif"] else "sans-serif"
    plt.rcParams.update({
            "font.family" : family,
            "font.serif": font,
            "font.size" : fontsize
        })

def unique_nodes_from(pdf, src_name, dst_name):
    """
    Get the nodes of the pdf
    """
    pdf = pdf.loc[:, [src_name, dst_name]]
    return np.unique(pdf.to_numpy().ravel('K'))

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
    elif ref_model.objective == "NodeEmb":
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
        meas = "ens_mean_lbci_ubci_" + meas
    
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
def save_fig(fig, full_path = None, save = True, dpi = 300):
    if save:
        import os
        dir_ = os.path.dirname(full_path)
        os.makedirs(dir_, exist_ok = True)
        
        fig.savefig(full_path, dpi = dpi, bbox_inches="tight", format = full_path[-3:])

def save_dict(full_path, dict_, save = True):
    """
    save dictionary at full_path
    """
    
    from pickle import dump
    if save:
        with open(full_path, 'wb') as f:
            dump(dict_, f)

def load_dict(full_path):
    """
    save dictionary at full_path
    """
    
    from pickle import load	
    
    return load(open(full_path, 'rb'))

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

def max_sampled_graph_idx(dir_, num_samples = None):
    """
    Find the maximum idx of the already sampled graph.
    Therefore if the n_sampels < ens_max_idx, no need to sample
    Return:
        - -1: no ensemble folder exists,
        - -1: the folder is empty,
        - ens_max_idx-1: the folder with the maximum index contains no files.
    """
    import os
    import re
    # If you want less than already sampled, just take the min
    min_wrt_num_samples = lambda x: np.min([x, num_samples - 1])

    if os.path.exists(dir_):
        filenames = next(os.walk(dir_))[1]
        if len(filenames) > 0:
            # substitute every non digit (\D) with '' (nothing)
            indices = [int(re.sub(pattern = r'\D', repl = '', string = x)) for x in filenames]
            argmax_idx = np.argmax(indices)
            ens_max_idx = indices[argmax_idx]
            # Find the folder name corresponding to the max index
            max_idx_folder = filenames[argmax_idx] #[f for f in filenames if int(re.sub(r'\D', '', f)) == ens_max_idx][0]
            max_idx_folder_path = os.path.join(dir_, max_idx_folder)
            
            # Check if the folder with the maximum index contains any files
            if os.path.exists(max_idx_folder_path) and len(os.listdir(max_idx_folder_path)) > 0:
                return min_wrt_num_samples(ens_max_idx)
            
            # otherwise return the previous (-1)
            return min_wrt_num_samples(ens_max_idx)-1
        else:
            return -1
    else:
        # this helps to create the first graph_0 in the folder. Otherwise, i - max_graph_idx >= 0:
        return -1

def set_model_pr_on_I(model, gI, vec_meas=["_pr"]):
    """
    Set measurement vectors (e.g., page-rank) on internal nodes.
    For each measurement in vec_meas, fills:
      - model.{meas}_on_I, model.{meas}_std_on_I
      - model.{meas}_on_I_rank, model.{meas}_on_I_desc, model.{meas}_on_I_std_desc
    """
    mod_dict = model.__dict__
    idx = gI.internal_nodes
    argsort_desc = lambda x: np.argsort(x)[::-1]

    if not hasattr(model, "_pr_on_I"):
        for meas in vec_meas:
            # Set measurement and std on internal nodes
            values_on_I = mod_dict[meas][idx]
            std_on_I = mod_dict[f"{meas}_std"][idx]

            # Rank indices (descending)
            rank_on_I = argsort_desc(values_on_I)

            # Sorted values and std
            on_I_desc = values_on_I[rank_on_I]
            on_I_std_desc = std_on_I[rank_on_I]

            # Assign to model
            mod_dict[f"{meas}_on_I"] = values_on_I
            mod_dict[f"{meas}_std_on_I"] = std_on_I

            # create new variables for vec_meas ranked from top to bottom
            mod_dict[f"{meas}_on_I_rank"] = rank_on_I
            mod_dict[f"{meas}_on_I_desc"] = on_I_desc
            mod_dict[f"{meas}_on_I_std_desc"] = on_I_std_desc

def tot_rel_err(x, y, ord = 1):

    return np.linalg.norm(signed_rel_err(x, y), ord = ord)

def signed_rel_err(x, y):
    """ 
    Relative error among each element of x and y, with modules. 
    Returns: (x-y) / y
    """
    return (x-y) / y

def rel_err(x, y):
    if x.shape != y.shape:
        ValueError(f"Shapes are not the same: x.shape {x.shape}, y.shape {y.shape}")
    return abs(signed_rel_err(x, y))

def rel_err_norm(x, y, ord = 1):
    """ Relative error between x and y. It returns a scalar """
    return np.linalg.norm(x - y, ord = ord) / np.linalg.norm(y, ord = ord)

def max_sqerr_rel_err(x, y, only_positive = True):
    if only_positive:
        xidx, yidx = x>0, y>0
        xidx_union_yidx = xidx + yidx
        x = x[xidx_union_yidx]
        y = y[xidx_union_yidx]
    rel_error = rel_err(x, y)
    
    return np.max(rel_error), np.linalg.norm(rel_error)

def pmatrix_vectorized(g, param, unsampled_vI, edges_in_p = False):

    """
    To chekc the results via numba, one can use this function to calculate them directly on the pmatrix.
    As the number of involved nodes increases, it won't be possible to build the pmatrix

    Return: pmatrix, matrix of internal edges
    """
   
    out_in_strength_matrix = g._out_strength.reshape(-1, 1) @ g._in_strength.reshape(1, -1)
    # print(f'-out_in_strength_matrix: {out_in_strength_matrix.shape}',)

    unsampled_vI = unsampled_vI.astype(bool)
    
    p = -np.expm1(-param * out_in_strength_matrix)

    # set to 1 the probabilities for the fixed internal nodes
    unsampled_pairs = unsampled_vI.reshape(-1, 1) @ unsampled_vI.reshape(1, -1)
    
    observed_value_unsampled_pairs = g.adjacency_matrix().todense()[unsampled_pairs]
    # print(f'-np.sum(observed_value_unsampled_pairs): {np.sum(observed_value_unsampled_pairs)}',)
    
    # if = 0, return the "stochastic" part, i.e. without the frozen edges
    # print(f'-edges_in_p: {edges_in_p}',)
    if edges_in_p:
        p[unsampled_pairs] = observed_value_unsampled_pairs
    else:
        p[unsampled_pairs] = 0

    unsampled_matrix = np.zeros_like(out_in_strength_matrix)
    unsampled_matrix[unsampled_pairs] = observed_value_unsampled_pairs
    

    # remove diagonal
    p *= 1-np.eye(p.shape[0])
    unsampled_matrix *= 1-np.eye(p.shape[0])

    return p, unsampled_matrix

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

def check_cpu_gpu_with_torch():
    import os
    import torch as tc

    print("\n-Logical CPUs:", os.cpu_count())

    print('-GPUs in use:')
    if tc.cuda.is_available():
        cc_cores_per_SM_dict = [
            [30, 192],
            [32, 192],
            [35, 192],
            [37, 192],
            [50, 128],
            [52, 128],
            [53, 128],
            [60,  64],
            [61, 128],
            [62, 128],
            [70,  64],
            [72,  64],
            [75,  64],
            [80,  64],
            [86, 128],
            [87, 128],
            [89, 128],
            [90, 128],
            [-1, -1]
        ]
        cc_cores_per_SM_dict = {k: v for k, v in cc_cores_per_SM_dict}        

        num_gpus = tc.cuda.device_count()
        print(f"-Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            name = tc.cuda.get_device_name(i)
            cc_major, cc_minor = tc.cuda.get_device_capability(i)
            majmin_cc = "".join((str(cc_major), str(cc_minor)))
            cores_per_sm = cc_cores_per_SM_dict.get(int(majmin_cc))

            dev_mp_count = tc.cuda.get_device_properties(i).multi_processor_count
            total_cores = cores_per_sm*dev_mp_count


            print(f"  GPU {i}: {name}")
            print(f"    - Compute Capability: {cc_major}{cc_minor}")
            print(f"    - Total Memory: {tc.cuda.get_device_properties(i).total_memory // (1024**2)} MB")
            print(f"    - Multiprocessors: {dev_mp_count}")
            # Optionally, you can add more properties if needed
            print(f"-Current CUDA device: {tc.cuda.current_device()}")
            print(f"-CUDA version: {tc.version.cuda}")
            print(f'-total_cores: {total_cores}',)
    else:
        print('-None')