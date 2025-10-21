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


    def topN_overlap_2_meas_fast(true_meas, est_meas, max_rank = None):
        """
        Calculate Jaccard similarity between ranked groups incrementally.
        
        Args:
            true_meas (np.ndarray): First measure (e.g. page rank values)
            est_meas (np.ndarray): Second measure (e.g. out-degrees)
            num_vertices (int): Optional early stopping point
            
        Returns:
            tuple: (range(1,N+1), overlap_values)
        """

        # Get ranked groups
        true_ranked = overlap_helpers.get_ranked_groups(true_meas)
        est_ranked = overlap_helpers.get_ranked_groups(est_meas)
        
        # print(f'-true_ranked: {true_ranked}',)
        # print(f'-est_ranked: {est_ranked}',)

        # Initialize empty sets and results
        topN_true = set()
        topN_est = set()
        overlaps = []
        
        # Define Jaccard similarity
        jacc_similarity = lambda A, B: len(A & B) / len(A | B)
        
        # Iterate through ranks
        if max_rank == None:
            max_rank = min(len(true_ranked), len(est_ranked))
        for N in range(max_rank):
            
            # Add nodes at current rank to sets
            topN_true.update(true_ranked[N])
            topN_est.update(est_ranked[N])
            
            # Calculate overlap
            overlap = jacc_similarity(topN_true, topN_est)
            overlaps.append(overlap)

            # print(f'\n-N: {N}',)
            # print(f'-topN_true: {topN_true}',)
            # print(f'-topN_est: {topN_est}',)
            # print(f'-overlap: {overlap}',)
                
        return range(1, len(overlaps) + 1), overlaps

    def topN_overlap_pr_out_in(g, gI_model = None, measures = ["_pr"]):
        """
        Calculate the topN_overlap between the out/in deg rankings.
        The function is abstract so refer to the Example for a better understanding of the passages.
        Indeed, to host every measure and class we did gI_model_dict = gI_model.__dict__. In the example, instead, there is a practical application for out_degree and gI
        gI_model: gI (internal) or g itgI_model

        Example:
        gI._topN_out_degree = topN_by_rank(gI._out_degree)
        gI._topN_overlap_out_degree = topN_overlap_g_pr_on_I(gI._topN_out_degree)
        """
        # if no specification, then ground truth is assumed to be passed
        if gI_model == None:
            gI_model = g
        
        g_pr_on_I = g._pr_on_I
        max_rank = g.__dict__.get("_topN_max_rank", None)
        
        # smart way of creating a not-existing var with changing name
        gI_model_dict = gI_model.__dict__
        
        # loop over the measures
        for meas in measures:
            var_name = "_topN_overlap"+meas
            gI_model_dict[var_name+"_range"], gI_model_dict[var_name] \
                                = overlap_helpers.topN_overlap_2_meas_fast(g_pr_on_I, gI_model_dict[meas], max_rank)


    def topN_overlap_pr(g, gI_model = None):
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
            
        return g.topN_overlap_pr_out_in(gI_model, pr_measure)
    
    def topN_overlap_pr_out_in_degree(g, gI_model = None):
        """
        Overlap btw the PR-ground-truth and the out/in degrees
        """

        # gI_model == None --> ground truth
        # gI_model.graph_kind == "intra" and gI_model.kind == "obs" --> gI (the internal network)
        isinternal = gI_model.graph_kind == "intra" and gI_model.kind == "obs" if gI_model is not None else None
        if gI_model == None or not isinternal:
            degree_measures = ["_out_degree_on_I", "_in_degree_on_I"]
        elif isinternal:
            degree_measures = ["_out_degree", "_in_degree"]

        return g.topN_overlap_pr_out_in(gI_model, degree_measures)

    def topN_overlap_pr_out_in_strengths(g):
        """
        Calculate the Overlap Between the PR of self and 
        """
        return g.topN_overlap_pr_out_in(measures = ["_out_strength_on_I", "_in_strength_on_I"])

    def out_in_degree_strength_on_I(g, gI, model = None):
        """
        Calculate the usefull measures for computing the overlap
        Note: model._pr_on_I was already computed
        """
        internal_nodes = gI.internal_nodes

        # calculate the strengths
        g._out_strength_on_I = g._out_strength[internal_nodes]
        g._in_strength_on_I = g._in_strength[internal_nodes]

        g._out_degree_on_I = g._out_degree[internal_nodes]
        g._in_degree_on_I = g._in_degree[internal_nodes]

        # calculate the degrees on I for model
        if model is not None:
            model._out_degree_on_I = model._out_degree[internal_nodes]
            model._in_degree_on_I = model._in_degree[internal_nodes]

    def topN_overlap_pr_deg_stre(g, gI, model):
        """
        g is treated as self
        """

        # overlap between the pr_on_I and out/in strengths/degree rankings
        g.topN_overlap_pr_out_in_strengths()
        g.topN_overlap_pr_out_in_degree()

        # internal graph out/in degree
        g.topN_overlap_pr_out_in_degree(gI)
        g.topN_overlap_pr_out_in_degree(model)