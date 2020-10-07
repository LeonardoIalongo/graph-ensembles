import numpy as np
from numba import jit


@jit(nopython=True,parallel=True)
def vector_fitness_prob_array_block_one_z(out_strength, in_strength, z, dict_sectors):
    """
    Function computing the Probability Matrix of the Cimi block model 
    with just one parameter controlling for the density.
    """
    
    N = len(np.unique(out_strength[:,0]))
    p = np.zeros((N, N), dtype=np.float64)
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        sect_node1 = int(dict_sectors[ind_out])
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            sect_node2 = int(dict_sectors[ind_in])
            if (ind_out != ind_in)&(sect_out==sect_node2)&(sect_in==sect_node1):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


@jit(forceobj=True,parallel=True)
def vector_fitness_link_prob_block_one_z(out_strength, in_strength, z, dict_sectors):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0
    
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        sect_node1 = int(dict_sectors[ind_out])
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            sect_node2 = int(dict_sectors[ind_in])
            if (ind_out != ind_in)&(sect_out==sect_node2)&(sect_in==sect_node1):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p