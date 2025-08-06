from .utils import *
import numpy as np
import pandas as pd
import os

def default_4_undirected_paper(dataset_name):
    ''' Default values for the first 2 undirected papers'''
    
    if dataset_name.startswith("Gleditsch"):
        id_code = "int"
        cg_method = "geo-dist"
        year = 2000
            
    elif dataset_name.startswith("ING"):
        # write here the naics code. If None, it will read the whole dataset with the grid_id names
        id_code = "naics_code"
        cg_method = "naics_code"
        year = 2022
    
    return id_code, cg_method, year

def flip_payer_beneficiary_columns(df):
	"""
    Flips the values between 'payer_...' and 'beneficiary_...' columns

    Finds columns starting with 'payer_' and checks if a corresponding
    'beneficiary_' column with the same suffix exists. 
	If both exist, the values in these two columns are swapped for every row.
	If not, an error arise.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the columns flipped where applicable.
                      Returns a copy of the original if no pairs are found.
    """

	# Extract all suffixes from payer and beneficiary columns
	# 
	suffixes = set()
	for col in df.columns:
		if col.startswith('payer_'):
			suffix = col.split('payer_', 1)[1]
			suffixes.add(suffix)
		elif col.startswith('beneficiary_'):
			suffix = col.split('beneficiary_', 1)[1]
			suffixes.add(suffix)
	
	# Swap values for each payer-beneficiary column pair
	for suffix in suffixes:
		payer_col = f'payer_{suffix}'
		beneficiary_col = f'beneficiary_{suffix}'
		if payer_col in df.columns and beneficiary_col in df.columns:
			df[payer_col], df[beneficiary_col] = df[beneficiary_col], df[payer_col]
	
	return df

def dataset_loader(name, corpkey = None, dataset_direction = "Undirected", id_code = "naics_code", cg_method = "naics_code", year = 2022, distance_matrix = None, lvl_to_nclust = None, max_num_entries = 0):

    # dataset name year and direction (nyd)
    dataset_nyd = f"{name}-{dataset_direction}"

    # if on dap, it will execute the try, otherwise if local the read_csv
    xgrid_time_xtrans_time = "xgrid_20240427_xtrans_20240424" if dataset_direction == "Directed" else "xgrid_20240404_xtrans_20240324"
    dataset_folder = f"{os.path.expanduser('~')}/Documents/Datasets/{dataset_nyd}"

    if dataset_nyd.startswith("Gleditsch"):
        pdtrans = pd.read_csv(dataset_folder + "/edges_int_pd.csv")
        print(f"\nReading from local source @ {dataset_folder + '/edges_int_pd.csv'}")
        
    elif dataset_nyd.startswith("ING"):

        # select the last two digis of the year
        last_digits = str(year)[2:]
        full_path = lambda dataset_folder: f"{dataset_folder}/{xgrid_time_xtrans_time}/pdtrans_no_rotw_gridSelfLoops_52559299_{id_code}.csv"
        
        # decide whether to go for COREALGOS directory or LOCAL one based on the corpkey finding
        if corpkey:
            dataset_folder = os.path.expanduser('~') + "/data/corealgos/rmilocco"
            print(f"\nReading from COREALGOS")# @ {full_path(dataset_folder)}")

        else:
            # the naics_code dataset is sorted (ascending) by payer and beneficiaries
            print(f"\nReading from local source")

        pdtrans = pd.read_csv(full_path(dataset_folder)).drop(columns=["nrofpayments"])

    if "payer_naics_desc" in pdtrans.columns:
        pdtrans.drop(["payer_naics_desc", "beneficiary_naics_desc"], axis = 1, inplace = True)

    if sum(pdtrans.iloc[:, 0] == pdtrans.iloc[:, 1]):
        idx_selfloops = pdtrans.loc[:, f"payer_{id_code}"] == pdtrans.iloc[:, f"beneficiary_{id_code}"]
        pdtrans = pdtrans[~idx_selfloops].reset_index(drop=True)
        print('-Removed the SelfLoops',)
        
    # COARSE-GRAINING METHODS   
    if cg_method.startswith(("geo-dist", "rand-dist")):
        pdtrans.columns = ["payer_int", "beneficiary_int", 'amount_euro']

        # define the diminishing rate of nodes from level to level + 1
        n_nodes_0 = nodes_from(pdtrans, id_code = 'int').size
        delta_n_clust = 30
        
        # assuming that a network with a number of nodes < lowest_n_nodes (e.g. 10) does not makes sense
        # find level l : N - l * r > c (N = n_nodes_0, r = delta_n_clust, c = lowest_n_nodes)
        lowest_n_nodes = 0
        n_clust_seq = [n_nodes_0 - l * delta_n_clust for l in range(1 + (n_nodes_0 - lowest_n_nodes) // delta_n_clust )]
        
        # num of levels is how many n_clust_seq before lowest_n_nodes
        total_levels = len(n_clust_seq)
        lvl_to_nclust = {k:v for k, v in enumerate(n_clust_seq)}

        if cg_method.startswith(("geo-dist")):
            # load the distance matrix, i.e. the geographical distance ones
            distance_matrix = pd.read_csv(dataset_folder + f"/geo_dist.csv", index_col = 0)
        

    elif cg_method.startswith("naics"):
        # set the total_levels to the numb of times we can cut the last digits of the naics_code, i.e. 5 (6,5,4,3,2)
        total_levels = len(str(pdtrans[f"payer_naics_code"].iloc[0])) - 1

    elif cg_method.startswith("random"):
        total_levels = 1
    
    if not (max_num_entries == None or max_num_entries == False):
        np.random.seed(0)
        max_num_entries = int(max_num_entries)
        random_entries = np.random.choice(len(pdtrans), size = max_num_entries, replace = False)
        pdtrans = pdtrans.iloc[random_entries]

    if dataset_direction == "Directed":
        # perform a payer_id_code <-> ben_id_code substitution to retain the supply-chain network
        pdtrans = flip_payer_beneficiary_columns(pdtrans)
        
        # sort the values with respect the columns before the ``amount euro''
        pdtrans = pdtrans.sort_values(by=pdtrans.columns[:-1].to_list())\
                        .reset_index(drop=True)

    kwargs = {
            "name" : name,
            "dataset_name" : dataset_nyd,
            # "dataset_direction" : dataset_direction,
            "level" : 0,
            "pdtrans" : pdtrans.copy(),
            "corpkey" : corpkey,
            "id_code" : id_code,
            "year" : year,
            "cg_method" : cg_method, # needed for defining the directory
            # "micro_id_codes" : micro_id_codes,
            # these 2 lines works only for random partitioning, i.e. "id_code" : "grid_id"
            "distance_matrix" : distance_matrix, "lvl_to_nclust" : lvl_to_nclust,
            "dataset_folder" : dataset_folder,
            }

    return pdtrans, kwargs, total_levels