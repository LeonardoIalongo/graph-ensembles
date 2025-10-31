class overlap_helpers:

    @staticmethod
    def get_ranked_groups(arr):
        """
        Group indices by their values in descending order.
        
        Args:
            arr: Input array
            
        Returns:
            List of lists containing indices grouped by unique values in descending order
        """
        import numpy as np
        # Get unique values in descending order
        unique_vals = sorted(set(arr), reverse=True)
        # Group indices that share the same value
        return [np.where(arr == val)[0] for val in unique_vals]

    def similarity(true, est, type_ = "jaccard"):
        
        intersection = len(true & est)
        if type_ == "jaccard":
            return intersection / len(true | est)
        elif type_ == "fraction":
            return intersection / len(est)

    @staticmethod
    def topN_overlap_2_meas_fast(true_meas, est_meas, max_pos = None):
        """
        Calculate Jaccard similarity between ranked groups incrementally.
        
        Args:
            true_meas (np.ndarray): First measure (e.g. page rank values)
            est_meas (np.ndarray): Second measure (e.g. out-degrees)
            num_vertices (int): Optional early stopping point
            
        Returns:
            tuple: (range(1,N+1), overlap_values)
        """
        import numpy as np

        # Get ranked groups
        num_vertices = len(true_meas)
        true_ranked = overlap_helpers.get_ranked_groups(true_meas)
        est_ranked = overlap_helpers.get_ranked_groups(est_meas)
        
        # print(f'-true_ranked: {true_ranked}')
        # print(f'-est_ranked: {est_ranked}')
        # print(f'-len(true_ranked): {len(true_ranked)}',)
        # print(f'-len(est_ranked): {len(est_ranked)}',)

        # Initialize empty sets and results
        topN_true = set()
        topN_est = set()
        overlaps = []
        overlaps_range = []
        start_true_ranked = 0
        
        for est_pos in range(len(est_ranked)):
            # update the set of topN_est
            topN_est.update(est_ranked[est_pos])
            
            # print(f'\n-est_pos: {est_pos}',)

            # Add nodes at current rank to sets
            if len(topN_true) < num_vertices:
                end_true_ranked = start_true_ranked + len(est_ranked[est_pos])
                
                # update the topN_true with the positions equal to the number of new nodes in est_ranked[est_pos]
                new_true_meas = np.concatenate(true_ranked[start_true_ranked:end_true_ranked])
                topN_true.update(new_true_meas)

                # update the start_true_ranked
                start_true_ranked = end_true_ranked
        
            # print(f'-topN_true: {topN_true}',)
            # print(f'-topN_est: {topN_est}',)
            
            # Calculate overlap
            overlap = overlap_helpers.similarity(topN_true, topN_est, "fraction")
            overlaps.append(overlap)
            overlaps_range.append(len(topN_est))
    
        return overlaps_range, overlaps

        # Iterate through ranks
        # set it to min if you want to remove the straight line at the end (inclusion rather than similarity)
        # if max_pos == None:
        #     max_pos = max(len(true_ranked), len(est_ranked)) 
        # for N in range(max_pos):
            
        #     # Add nodes at current rank to sets
        #     if N < len(true_ranked):
        #         topN_true.update(true_ranked[N])
        #     if N < len(est_ranked):
        #         topN_est.update(est_ranked[N])
            
        #     # Calculate overlap
        #     overlap = overlap_helpers.similarity(topN_true, topN_est)
        #     overlaps.append(overlap)
        #     overlaps_range.append(len(topN_est))

            # print(f'\n-N: {N}',)
            # print(f'-topN_true: {topN_true}',)
            # print(f'-topN_est: {topN_est}',)
            # print(f'-overlap: {overlap}',)
                
        # return range(1, len(overlaps) + 1), overlaps

    def topN_overlap_pr_out_in(g, gI_model = None, measures = ["_pr"], recompute = False):
        """
        Calculate the topN_overlap between the out/in deg rankings.
        The function is abstract so refer to the Example for a better understanding of the passages.
        Indeed, to host every measure and class we did gI_model_vars = gI_model.__dict__. In the example, instead, there is a practical application for out_degree and gI
        gI_model: gI (internal) or g

        Example:
        gI._topN_out_degree = topN_by_rank(gI._out_degree)
        gI._topN_overlap_out_degree = topN_overlap_g_pr_on_I(gI._topN_out_degree)
        """
        import os
        from graph_ensembles import utils

        # if no specification, then ground truth is assumed to be passed
        if gI_model == None:
            gI_model = g
            base_folder = g.vars_dir_vsplit
        else:
            base_folder = gI_model.vars_dir

        # smart way of creating a not-existing var with changing name
        gI_model_vars = gI_model.__dict__

        if len(measures) == 2:
            fname = base_folder + "/topN_overlap_out_in_" + measures[0].strip("_out") + "_range.pkl"
        else:
            fname = base_folder + "/topN_overlap" + measures[0] + "_range.pkl"
        
        if gI_model.kind == "exp":
            kind = gI_model.fit_method
        else:
            kind = gI_model.graph_kind

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if not os.path.exists(fname) or recompute:

            g_pr_on_I = g._pr_on_I
            max_pos = g.__dict__.get("_topN_max_pos", None)
            
            # loop over the measures and select only the new ones to be saved
            meas_dict = {}
            for meas in measures:
                print(f'-Computing topN overlap PR VS {meas} for {gI_model.name}-{kind}', )
                var_name = "_topN_overlap"+meas
                gI_model_vars[var_name+"_range"], gI_model_vars[var_name] \
                                    = overlap_helpers.topN_overlap_2_meas_fast(g_pr_on_I, gI_model_vars[meas], max_pos)
                meas_dict[var_name+"_range"], meas_dict[var_name] = gI_model_vars[var_name+"_range"], gI_model_vars[var_name]
            
            # print(f'-To fname: {fname}',)
            utils.save_dict(fname, meas_dict)
        
        else:
            print(f'-Loading topN overlap PR VS {measures} for {gI_model.name}-{kind}')
            # print(f'-From fname: {fname}',)
            meas_dict = utils.load_dict(fname)
            gI_model_vars.update(meas_dict)


    def topN_overlap_pr(g, gI_model = None, recompute = False):
        """
        Overlap btw the PR-ground-truth and the out/in degrees
        """

        # if gI_model == None, then gI_model must equal g. So, g = gI_model
        if gI_model == None:
            gI_model = g

        if gI_model.graph_kind == "intra" and gI_model.kind == "obs":
            pr_measure = ["_pr"]
        else:
            pr_measure = ["_pr_on_I"]
            
        return g.topN_overlap_pr_out_in(gI_model, pr_measure, recompute)
    
    def topN_overlap_pr_out_in_degree(g, gI_model = None, recompute = False):
        """
        Overlap btw the PR-ground-truth and the out/in degrees
        """

        
        # ground truth
        if gI_model == None:
            degree_measures = ["_out_degree_on_I", "_in_degree_on_I"]
        
        # internal and observed
        elif gI_model.graph_kind == "intra" and gI_model.kind == "obs": # isinternal
            degree_measures = ["_out_degree", "_in_degree"]
        
        # model case
        else:
            degree_measures = ["_out_degree_on_I", "_in_degree_on_I"]

        return g.topN_overlap_pr_out_in(gI_model, degree_measures, recompute)

    def topN_overlap_pr_out_in_strengths(g, recompute = False):
        """
        Calculate the Overlap Between the PR of self and 
        """
        return g.topN_overlap_pr_out_in(measures = ["_out_strength_on_I", "_in_strength_on_I"], recompute = recompute)

    def out_in_degree_strength_on_I(g, gI, model = None):
        """
        Calculate the usefull measures for computing the overlap
        Note: model._pr_on_I was already computed
        """
        # calculate the strengths
        g._out_strength_on_I = gI._out_strength
        g._in_strength_on_I = gI._in_strength
        
        internal_nodes = gI.internal_nodes

        # calculate the strengths
        # g._out_strength_on_I = g._out_strength[internal_nodes]
        # g._in_strength_on_I = g._in_strength[internal_nodes]
        
        # the degrees
        g._out_degree_on_I = g._out_degree[internal_nodes]
        g._in_degree_on_I = g._in_degree[internal_nodes]

        # calculate the degrees on I for model
        if model is not None:
            model._out_degree_on_I = model._out_degree[internal_nodes]
            model._in_degree_on_I = model._in_degree[internal_nodes]

    def topN_overlap_pr_deg_stre(g, gI, model, recompute = False):
        """
        g is treated as self
        """
        # set a new var_dirs for g
        from os.path import dirname as path_dirname
        dirname = lambda dir_: path_dirname(dir_)
        g.vars_dir_vsplit = dirname(dirname(gI.vars_dir)) + "/full"
        
        # overlap between the pr_on_I and out/in strengths/degree rankings
        g.topN_overlap_pr_out_in_strengths()
        g.topN_overlap_pr_out_in_degree()

        # internal graph out/in degree
        g.topN_overlap_pr_out_in_degree(gI)
        g.topN_overlap_pr_out_in_degree(model)