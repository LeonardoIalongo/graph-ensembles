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

def _set_mpl_params(font = "Times New Roman", fontsize = '23'):
    """
    Set latex if needed and fontsize
    """
    family = "serif" if font in ["Times New Roman"] else "sans-serif"
    plt.rcParams.update({
            "font.family" : family,
            "font.serif": font,
            "font.size" : fontsize
        })

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
def save_fig(fig, full_path = None, save = True):
    if save:
        import os
        dir_ = os.path.dirname(full_path)
        os.makedirs(dir_, exist_ok = True)
        
        fig.savefig(full_path, bbox_inches="tight")

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

def set_ivec_on_I(g, gI):
    """ 
    Set the ivec (e.g. page-rank) on the internal nodes.
    Fills: 
    1) g.ivec_on_I, gI.ivec
    2) g.rank_on_I, gI.rank
    """
    
    # prepare the meas over g and gI
    gI.idx_intnode_on_full = list(map(lambda x: g.id_dict.get(x), gI.id_dict))
    g.ivec_on_I = g.ivec[gI.idx_intnode_on_full]

    # obtain the idx of ranked (descending) meas
    inv_argsort = lambda x: np.argsort(x)[::-1]
    g.rank_on_I = inv_argsort(g.ivec_on_I)
    gI.rank = inv_argsort(gI.ivec)

    # obtain the descending ivec on I 
    g.ivec_desc_on_I = g.ivec_on_I[g.rank_on_I]
    gI.ivec_desc = gI.ivec[gI.rank]

    # save the variables on gI
    gI.save_vars(name = "graph")

def set_model_ivec_on_I(self, gI):
    """ 
    Set the ivec (e.g. page-rank) on the internal nodes.
    Fills: 
    1) model.ivec_on_I, model.ivec_std_on_I
    2) model.rank_on_I
    """

    self.ivec_on_I = self.ivec[gI.idx_intnode_on_full]
    self.ivec_std_on_I = self.ivec_std[gI.idx_intnode_on_full]

    # obtain the idx of ranked (descending) meas
    inv_argsort = lambda x: np.argsort(x)[::-1]
    self.rank_on_I = inv_argsort(self.ivec_on_I)
    self.ivec_desc_on_I = self.ivec_on_I[self.rank_on_I]
    self.ivec_std_desc_on_I = self.ivec_std_on_I[self.rank_on_I]

def signed_rel_err(x, y):
    """ 
    Relative error among each element of x and y, with modules. 
    Returns: (x-y) / y
    """
    return (x-y) / y

def rel_err_norm(x, y, ord = 1):
    """ Relative error between x and y. It returns a scalar """
    return np.linalg.norm(x - y, ord = ord) / np.linalg.norm(y, ord = ord)

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