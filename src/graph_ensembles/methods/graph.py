import numpy as np
from numba import jit


# --------------- GRAPH METHODS ---------------
def get_num_bytes(num_items):
    """ Determine the number of bytes needed for storing ids for num_items."""
    return int(max(2**np.ceil(np.log2(np.log2(num_items + 1)/8)), 1))


@jit(nopython=True)
def check_unique_edges(e):
    """ Check that the edges are not repeated in the sorted edge list."""
    for i in np.arange(len(e)-1):
        if (e[i].src == e[i+1].src) and (e[i].dst == e[i+1].dst):
            assert False, 'There are repeated edges.'


@jit(nopython=True)
def check_unique_labelled_edges(e):
    """ Check that the edges are not repeated in the sorted edge list."""
    for i in np.arange(len(e)-1):
        if ((e[i].label == e[i+1].label) and
           (e[i].src == e[i+1].src) and
           (e[i].dst == e[i+1].dst)):
            assert False, 'There are repeated edges.'


def generate_id_dict(v, id_col):
    """ Return id dictionary. """
    id_dict = {}
    rep_msg = 'There is at least one repeated id in the vertex dataframe.'

    if isinstance(id_col, list):
        if len(id_col) > 1:
            # Id is a tuple
            i = 0
            for x in v[id_col].itertuples(index=False):
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        elif len(id_col) == 1:
            # Extract series
            i = 0
            for x in v[id_col[0]]:
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        else:
            # No column passed
            raise ValueError('At least one id column must be given.')

    elif isinstance(id_col, str):
        # Extract series
        i = 0
        for x in v[id_col]:
            if x in id_dict:
                raise Exception(rep_msg)
            else:
                id_dict[x] = i
                i += 1

    else:
        raise ValueError('id_col must be string or list of strings.')

    return id_dict


def generate_label_dict(e, label):
    """ Return id dictionary. """
    label_dict = {}

    if isinstance(label, list):
        if len(label) > 1:
            # Id is a tuple
            i = 0
            for x in e[label].itertuples(index=False):
                if x in label_dict:
                    pass
                else:
                    label_dict[x] = i
                    i += 1

        elif len(label) == 1:
            # Extract series
            i = 0
            for x in e[label[0]]:
                if x in label_dict:
                    pass
                else:
                    label_dict[x] = i
                    i += 1

        else:
            # No column passed
            raise ValueError('At least one label column must be given.')

    elif isinstance(label, str):
        # Extract series
        i = 0
        for x in e[label]:
            if x in label_dict:
                pass
            else:
                label_dict[x] = i
                i += 1

    else:
        raise ValueError('edge_label must be string or list of strings.')

    return label_dict


@jit(nopython=True)
def compute_num_edges_by_label(e, num_lbl):
    edges = np.zeros(num_lbl, dtype=np.int64)
    for n in range(len(e)):
        edges[e[n].label] += 1

    return edges


@jit(nopython=True)
def compute_tot_weight_by_label(e, num_lbl):
    weight = np.zeros(num_lbl, dtype=np.float64)
    for n in range(len(e)):
        weight[e[n].label] += e[n].weight

    return weight


@jit(nopython=True)
def compute_degree(e, num_v):
    d = np.zeros(num_v, dtype=np.int64)
    s = set()
    for n in range(len(e)):
        i = e[n].src
        j = e[n].dst
        if i <= j:
            pair = (i, j)
        else:
            pair = (j, i)

        if pair not in s:
            s.add(pair)
            d[i] += 1
            d[j] += 1

    return d


@jit(nopython=True)
def compute_degree_by_label(e):
    lv_dict = {}
    i = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in lv_dict:
            lv_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in lv_dict:
            lv_dict[pair_d] = i
            i += 1

    d = np.zeros((len(lv_dict), 3), dtype=np.uint64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = lv_dict[tmp]
        d[ind, 0:2] = tmp
        d[ind, 2] += 1

        tmp = (e[n].label, e[n].dst)
        ind = lv_dict[tmp]
        d[ind, 0:2] = tmp
        d[ind, 2] += 1

    return d, lv_dict


@jit(nopython=True)
def compute_in_out_degree(e, num_v):
    d_out = np.zeros(num_v, dtype=np.int64)
    d_in = np.zeros(num_v, dtype=np.int64)
    for n in range(len(e)):
        d_out[e[n].src] += 1
        d_in[e[n].dst] += 1

    return d_out, d_in


@jit(nopython=True)
def compute_in_out_degree_labelled(e, num_v):
    d_out = np.zeros(num_v, dtype=np.int64)
    d_in = np.zeros(num_v, dtype=np.int64)
    s = set()
    for n in range(len(e)):
        i = e[n].src
        j = e[n].dst
        pair = (i, j)
        if pair not in s:
            s.add(pair)
            d_out[i] += 1
            d_in[j] += 1

    return d_out, d_in


@jit(nopython=True)
def compute_in_out_degree_by_label(e):
    out_dict = {}
    in_dict = {}
    i = 0
    j = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in out_dict:
            out_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in in_dict:
            in_dict[pair_d] = j
            j += 1

    d_out = np.zeros((len(out_dict), 3), dtype=np.uint64)
    d_in = np.zeros((len(in_dict), 3), dtype=np.uint64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = out_dict[tmp]
        d_out[ind, 0:2] = tmp
        d_out[ind, 2] += 1

        tmp = (e[n].label, e[n].dst)
        ind = in_dict[tmp]
        d_in[ind, 0:2] = tmp
        d_in[ind, 2] += 1

    return d_out, d_in, out_dict, in_dict


@jit(nopython=True)
def compute_strength(e, num_v):
    s = np.zeros(num_v, dtype=np.float64)

    for n in range(len(e)):
        s[e[n].src] += e[n].weight
        s[e[n].dst] += e[n].weight

    return s


@jit(nopython=True)
def compute_strength_by_label(e):
    lv_dict = {}
    i = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in lv_dict:
            lv_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in lv_dict:
            lv_dict[pair_d] = i
            i += 1

    s = np.zeros((len(lv_dict), 3), dtype=np.float64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = lv_dict[tmp]
        s[ind, 0:2] = tmp
        s[ind, 2] += e[n].weight

        tmp = (e[n].label, e[n].dst)
        ind = lv_dict[tmp]
        s[ind, 0:2] = tmp
        s[ind, 2] += e[n].weight

    return s, lv_dict


@jit(nopython=True)
def compute_in_out_strength(e, num_v):
    s_out = np.zeros(num_v, dtype=np.float64)
    s_in = np.zeros(num_v, dtype=np.float64)
    for n in range(len(e)):
        s_out[e[n].src] += e[n].weight
        s_in[e[n].dst] += e[n].weight

    return s_out, s_in


@jit(nopython=True)
def compute_in_out_strength_by_label(e):
    out_dict = {}
    in_dict = {}
    i = 0
    j = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in out_dict:
            out_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in in_dict:
            in_dict[pair_d] = j
            j += 1

    s_out = np.zeros((len(out_dict), 3), dtype=np.float64)
    s_in = np.zeros((len(in_dict), 3), dtype=np.float64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = out_dict[tmp]
        s_out[ind, 0] = e[n].label
        s_out[ind, 1] = e[n].src
        s_out[ind, 2] += e[n].weight

        tmp = (e[n].label, e[n].dst)
        ind = in_dict[tmp]
        s_in[ind, 0] = e[n].label
        s_in[ind, 1] = e[n].dst
        s_in[ind, 2] += e[n].weight

    return s_out, s_in, out_dict, in_dict
