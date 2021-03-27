""" This module contains any function that operates on the sGraph or
    GraphEnsemble classes or on its attributes.
"""

from scipy.sparse import coo_matrix


def to_sparse(coo_arr, shape, kind='coo', i_col=0, j_col=1, data_col=2):
    """ Convert to a sparse matrix the coordinate array passed.
    """
    mat = coo_matrix((coo_arr[:, data_col],
                     (coo_arr[:, i_col], coo_arr[:, j_col])),
                     shape=shape)
    if kind == 'coo':
        return mat
    else:
        return mat.asformat(kind)
