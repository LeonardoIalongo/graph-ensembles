""" This module contains any function that operates on the sGraph or
    GraphEnsemble classes or on its attributes.
"""

import graph_ensembles as ge
import numpy as np
from scipy.sparse import coo_matrix
from numpy.lib.recfunctions import rec_append_fields as append_fields
import warnings


def to_sparse(coo_arr, shape, kind='coo', i_col=0, j_col=1, data_col=2):
    """ Convert to a sparse matrix the coordinate array passed.
    """
    if isinstance(i_col, int):
        i = coo_arr[coo_arr.dtype.names[i_col]]
    elif isinstance(i_col, str):
        i = coo_arr[i_col]
    else:
        raise ValueError('i_col must be an int or a string.')

    if isinstance(j_col, int):
        j = coo_arr[coo_arr.dtype.names[j_col]]
    elif isinstance(j_col, str):
        j = coo_arr[j_col]
    else:
        raise ValueError('j_col must be an int or a string.')

    if isinstance(data_col, int):
        data = coo_arr[coo_arr.dtype.names[data_col]]
    elif isinstance(data_col, str):
        data = coo_arr[data_col]
    else:
        raise ValueError('data_col must be an int or a string.')

    mat = coo_matrix((data, (i, j)), shape=shape)

    if kind == 'coo':
        return mat
    else:
        return mat.asformat(kind)


def add_groups(g, group_dict):
    """ Add group info to a sGraph object, if already presents it raises a warning.
    """
    if hasattr(g, 'gv'):
        msg = 'Group info already present, will overwrite.'
        warnings.warn(msg, UserWarning)

    g.gv = ge.GroupVertexList()
    g.group_dict = group_dict
    g.num_groups = len(group_dict)
    num_bytes = ge.methods.get_num_bytes(g.num_groups)
    g.group_dtype = np.dtype('u' + str(num_bytes))
    if 'group' in g.v.dtype.names:
        g.v.group = group_dict
    else:
        g.v = append_fields(g.v, 'group', group_dict)
