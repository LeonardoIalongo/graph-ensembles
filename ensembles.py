""" This module defines the functions that allow for the construction of 
network ensembles from partial information. They can be used for 
reconstruction, filtering or pattern detection among others. """

import numpy as np 
import pandas as pd 


def get_strenghts(edges, vertices, group_col = None):
    """ Returns the strength sequence for the given network. If a group is 
    given then a in-strength value is returned for each group. The 
    out-strength is always one. """

    out_strenght = edges.groupby(['src'], as_index=False).agg(
        {'weight': sum}).rename(
        columns={"weight": "out_strength", "src": "id"})

    if group_col is None:
        in_strength = edges.groupby(['dst'], as_index=False).agg(
        {'weight': sum}).rename(
        columns={"weight": "in_strength", "dst": "id"})
    else:
        in_strength = edges.groupby(['dst'], as_index=False).agg(
        {'weight': sum}).rename(
        columns={"weight": "in_strength", "dst": "id"})

    strenght = out_strenght.join(in_strength.set_index('id'), 
        on='id', how='outer')

    return strenght.fillna(0)
