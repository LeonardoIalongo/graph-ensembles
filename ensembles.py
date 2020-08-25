""" This module defines the functions that allow for the construction of 
network ensembles from partial information. They can be used for 
reconstruction, filtering or pattern detection among others. """

import numpy as np 
import pandas as pd 


def get_strenghts(edges, vertices, group_col=None, group_dir='in' ):
    """ Returns the strength sequence for the given network. If a group is 
    given then it returns a vector for each strength where each element is 
    the strength related to each group. """

    if group_col is None:
        # If no group is specified return total strength
        out_strenght = edges.groupby(['src'], as_index=False).agg(
            {'weight': sum}).rename(
            columns={"weight": "out_strength", "src": "id"})

        in_strength = edges.groupby(['dst'], as_index=False).agg(
        {'weight': sum}).rename(
        columns={"weight": "in_strength", "dst": "id"})

        strength = out_strenght.join(
            in_strength.set_index('id'), on='id', how='outer').fillna(0)
    else:
        if group_dir in ['out', 'all']:
            # Get group of dst edge
            temp = edges.join(vertices.set_index('id'), on='dst', how='left')
            out_strenght = temp.groupby(['src', group_col],
                as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "out_strength", "src": "id"})
        else:
            out_strenght = edges.groupby(['src'], as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "out_strength", "src": "id"})

        if group_dir in ['in', 'all']:
            # Get group of src edge
            temp = edges.join(vertices.set_index('id'), on='src', how='left')
            in_strength = temp.groupby(['dst', group_col], 
                as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "in_strength", "dst": "id"})
        else:
            in_strength = edges.groupby(['dst'], as_index=False).agg(
                {'weight': sum}).rename(
                columns={"weight": "in_strength", "dst": "id"})

        strength = out_strenght.join(
            in_strength.set_index('id'), on='id', how='outer')

    
    return strength

