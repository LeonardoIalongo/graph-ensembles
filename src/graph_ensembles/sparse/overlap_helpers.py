class overlap_helpers:
    
    def topN_by_rank(self, arr):
        """
        Return a list containing the topN@K, where K is the placement in the rank.
        E.g. K = 3, provides the 1st, 2nd and 3rd classified nodes
        """
        import numpy as np

        # arr: 1D numpy array of values
        uniq_vals = np.unique(arr)[::-1]  # descending order
        return [np.where(arr >= val)[0] for val in uniq_vals]
    
    @staticmethod
    def topN_overlap_2_meas(true_meas, est_meas):
        """
        Returns 
        1) the overlap at topN among the true_meas and est_meas
        2) the minum number of nodes involved in the operation to be used in the topN plotting
        """

        true_len, est_len = len(true_meas), len(est_meas)
        topN_overlap = []
        min_num_nodes = []

        # range over the maximum placement in the rankings
        for l in range(max(true_len, est_len)):

            # i and j are clipped not to exceed its length
            i,j = min(l, true_len-1), min(l, est_len-1)
            true_meas_i, est_meas_j = set(true_meas[i]), set(est_meas[j])

            # compute the jaccard similarity as overlap
            overlap_ij = len(true_meas_i & est_meas_j) / len(true_meas_i.union(est_meas_j))
            topN_overlap.append(overlap_ij)

            # fill the min_num_nodes with the minimum among the true_meas_i and est_meas_j number of nodes
            min_num_nodes.append(min(len(true_meas_i), len(est_meas_j)))

        return min_num_nodes, topN_overlap

    def topN_overlap_g_pr_on_I(g, est_meas):
        
        if not hasattr(g, "_topN_pr_on_I"):
            g._topN_pr_on_I = g.topN_by_rank(g._pr_on_I)

        return overlap_helpers.topN_overlap_2_meas(g._topN_pr_on_I, est_meas)

    def topN_overlap_pr_out_in(g, gI_model, measures):
        """
        Calculate the topN_overlap between the out/in deg rankings.
        The function is abstract so refer to the Example for a better understanding of the passages.
        Indeed, to host every measure and class we did gI_model_dict = gI_model.__dict__. In the example, instead, there is a practical application for out_degree and gI
        gI_model: gI (internal) or g itgI_model

        Example:
        gI._topN_out_degree = topN_by_rank(gI._out_degree)
        gI._topN_overlap_out_degree = topN_overlap_g_pr_on_I(gI._topN_out_degree)
        """
        
        for meas in measures:
            gI_model_dict = gI_model.__dict__

            # calculate the topN of the respective meas
            gI_model_dict["_topN"+meas] = g.topN_by_rank(gI_model_dict[meas])

            # print(f'-gI_model_dict["_topN"+meas]: {gI_model_dict["_topN"+meas]}',)

            # calculate the overlal between the topN page-rank on I and the topN of the measure
            gI_model_dict["_topN_overlap"+meas+"_min_num_nodes"], gI_model_dict["_topN_overlap"+meas] = g.topN_overlap_g_pr_on_I(gI_model_dict["_topN"+meas])

    def topN_overlap_pr_out_in_degree(g, gI_model = None, internal_nodes = None):
        """
        Overlap btw the PR-ground-truth and the out/in degrees
        """

        # if gI_model == None, then gI_model must equal g. So, g = gI_model
        if gI_model == None:
            gI_model = g

        if gI_model.graph_kind == "intra" and gI_model.kind == "obs":
            degree_measures = ["_out_degree", "_in_degree"]
        else:
            degree_measures = ["_out_degree_on_I", "_in_degree_on_I"]

            if not hasattr(gI_model, "_out_degree_on_I"):
                gI_model._out_degree_on_I = gI_model._out_degree[internal_nodes]
                gI_model._in_degree_on_I = gI_model._in_degree[internal_nodes]

        return g.topN_overlap_pr_out_in(gI_model, degree_measures)

    def topN_overlap_pr_out_in_strengths(g, internal_nodes = None):
        """
        Calculate the Overlap Between the PR of self and 
        """

        if not hasattr(g, "_out_strength_on_I"):
            g._out_strength_on_I = g._out_strength[internal_nodes]
            g._in_strength_on_I = g._in_strength[internal_nodes]

        return g.topN_overlap_pr_out_in(g, measures = ["_out_strength_on_I", "_in_strength_on_I"])

    def topN_overlap_pr_deg_stre(g, gI, gI_model = None):

        if gI_model == None:
            gI_model = gI

        # topN_overlap btw pr and out/in deg
        internal_nodes = gI.internal_nodes
        
        # ground truth
        if not hasattr(g, "_topN_overlap_out_degree_on_I"):
            g.topN_overlap_pr_out_in_degree(internal_nodes=internal_nodes)
            g.topN_overlap_pr_out_in_strengths(internal_nodes=internal_nodes)

        # internal graph out/in degree
        g.topN_overlap_pr_out_in_degree(gI_model, internal_nodes=internal_nodes)
