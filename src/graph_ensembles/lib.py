""" This module contains any function that operates on the sGraph or
    GraphEnsemble classes or on its attributes.
"""

from scipy.sparse import coo_matrix


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
