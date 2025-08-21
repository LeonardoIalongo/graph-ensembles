from graph_ensembles.dependencies import *
from .. import dependencies as dep
from graph_ensembles.utils import load_meas, save_fig
from graph_ensembles import sparse as sp
import numpy as np
from .. import utils


def degree(g, ref_model, sum_model):
    """Plot the degree sequence in, out, and 11 for the observed network and the sum model"""
    full_path = sum_model.plots_dir + f"/topological_meas/degree/level{g.level}.pdf"

    model_label = "Summed" if sum_model.name.startswith("sum-") else "Fractioned"

    x0, y0, z0 = g._out_degree, ref_model._out_degree, sum_model._out_degree
    x1, y1, z1 = g._in_degree, ref_model._in_degree, sum_model._in_degree

    if not os.path.exists(full_path):
        fig, axs = plt.subplots(1, 2, figsize = (20,7))
        axis_scale = 'log'
        alpha = 0.5

        axs[0].scatter(x0, x0, marker = 'o', color = obs_color, label = 'Observed', alpha = alpha)
        axs[0].scatter(x0, y0, marker = '+', color = ref_model_color, label = ref_model.name.title(), alpha = alpha)
        # axs[0].scatter(x0, z0, marker = 'x', color = sum_model_color, label = model_label, alpha = alpha)
        axs[0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected',)
        axs[0].set_title("Out")
        axs[0].set_axisbelow(True)
        axs[0].grid(True)

        axs[1].scatter(x1, x1, marker = 'o', color = obs_color, label = 'Observed', alpha = alpha)
        axs[1].scatter(x1, y1, marker = '+', color = ref_model_color, label = ref_model.name.title(), alpha = alpha)
        # axs[1].scatter(x1, z1, marker = 'x', color = sum_model_color, label = model_label, alpha = alpha)
        axs[1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected',)
        axs[1].set_title("In")
        axs[1].set_axisbelow(True)
        axs[1].grid(True)

        # set legends
        leg = axs[0].legend(loc = 'lower right', markerscale=2.,)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        leg = axs[1].legend(loc = 'lower right', markerscale=2.,)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

        # axs[2].scatter(x2,x2, marker = 'o', color = obs_color, label = 'Observed')
        # axs[2].scatter(x2,y2, marker = '+', color = ref_model_color, label = ref_model.name)
        # axs[2].scatter(x2,z2, marker = 'x', color = sum_model_color, label = model_label)
        # axs[2].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected')
        # axs[2].set_ylim(np.min(0.5*x2[x2>0]), 1.2*np.max(x2))
        # axs[2].set_title("Reciprocated")
        # axs[2].legend()
        # axs[2].set_axisbelow(True)
        # axs[2].grid(True)

        save_fig(fig, full_path)

        plt.close()

def set_xylabels(ax, obs_meas, exp_meas, sum_meas, axis_scale = 'log'):
    from matplotlib import ticker
    x_min, x_max = np.min([obs_meas, exp_meas, sum_meas]), np.max([obs_meas, exp_meas, sum_meas])

    # if the difference lays one order of magnitude and the axis_scale is log, erase the minor ticks and place n_bins
    if (x_max - x_min) / x_min < 1 and axis_scale == 'log':
        locator = ticker.MaxNLocator(nbins=3)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())

def annd_IO(net, ref_model, sum_model):
    """Plot the annd in-in, out-out, in-out, out-in"""
    full_path = sum_model.plots_dir + f"/bin_meas_vs_deg/level{net.level}/annd.pdf"

    model_label = "Summed" if sum_model.name.startswith("sum-") else "Fractioned"

    create_numpy = lambda x: x[~np.isnan(x)]

    def filter_nans(x, y, z):
        # remove nans based on the ones which have more nans
        nans = np.isnan(x) | np.isnan(y) | np.isnan(z) | np.logical_or(x == 0, y == 0, z == 0)

        return x[~nans], y[~nans], z[~nans]

    x0, y0, z0 = filter_nans(net.anndoo, ref_model.anndoo, sum_model.anndoo)
    x1, y1, z1 = filter_nans(net.anndii, ref_model.anndii, sum_model.anndii)
    x2, y2, z2 = filter_nans(net.anndoi, ref_model.anndoi, sum_model.anndoi)
    x3, y3, z3 = filter_nans(net.anndio, ref_model.anndio, sum_model.anndio)

    if not os.path.exists(full_path):
        fig, axs = plt.subplots(2, 2, figsize = (15,10))
        axis_scale = 'log'
        axs[0,0].scatter(x0, x0, marker = 'o', color = obs_color, label = 'Observed')
        axs[0,0].scatter(x0, y0, marker = '+', color = ref_model_color, label = 'Expected')
        axs[0,0].scatter(x0, z0, marker = 'x', color = sum_model_color, label = model_label)
        axs[0,0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected',)
        set_xylabels(axs[0,0], x0, y0, z0)
        # axs[0,0].set_ylim(0.5*np.min(x0[x0>0]), 1.2*np.max(x0))
        axs[0,0].set_title("Out-Out")
        axs[0,0].legend()
        axs[0,0].set_axisbelow(True)
        axs[0,0].grid(True)

        axs[0,1].scatter(x1, x1, marker = 'o', color = obs_color, label = 'Observed')
        axs[0,1].scatter(x1, y1, marker = '+', color = ref_model_color, label = 'Expected')
        axs[0,1].scatter(x1, z1, marker = 'x', color = sum_model_color, label = model_label)
        axs[0,1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected',)
        # axs[0,1].set_ylim(50)
        axs[0,1].set_title("In-In")
        axs[0,1].legend()
        axs[0,1].set_axisbelow(True)
        axs[0,1].grid(True)

        axs[1,0].scatter(x2,x2, marker = 'o', color = obs_color, label = 'Observed')
        axs[1,0].scatter(x2,y2, marker = '+', color = ref_model_color, label = 'Expected')
        axs[1,0].scatter(x2,z2, marker = 'x', color = sum_model_color, label = model_label)
        axs[1,0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected')
        # axs[1,0].set_ylim(np.min(0.5*x2[x2>0]), 1.2*np.max(x2))
        axs[1,0].set_title("Out-In")
        axs[1,0].legend()
        axs[1,0].set_axisbelow(True)
        axs[1,0].grid(True)

        axs[1,1].scatter(x3,x3, marker = 'o', color = obs_color, label = 'Observed')
        axs[1,1].scatter(x3,y3, marker = '+', color = ref_model_color, label = 'Expected')
        axs[1,1].scatter(x3,z3, marker = 'x', color = sum_model_color, label = model_label)
        axs[1,1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Observed', ylabel = 'Expected')
        # axs[1,1].set_ylim(50)
        axs[1,1].set_title("In-Out")
        axs[1,1].legend()
        axs[1,1].set_axisbelow(True)
        axs[1,1].grid(True)


        if sum_model.name.startswith("sum-"):
            fig.suptitle(f"Degrees of the {sum_model.name} ones @ level {net.level}")
        else:
            fig.suptitle(f"Degrees of the {sum_model.name} : from {sum_model.top_level} (top level) --> {ref_model.level} (level)")

        save_fig(fig, full_path)

        plt.close()

def normalize(X):
    X_norm = tc.linalg.norm(X, axis = 1)[:, None]
    mask = X_norm != 0
    return tc.where(mask, X / X_norm, X)

def plots_rel_err_n_edges_across_levels(sum_model, model_names, total_levels, markers, colors, stripes_level = None):
    """
    Plot the relative error across the levels either for the summed and fined models
    NB: model_names must already carry "sum-" or "fine-" prefix
    """
    from utils import load_array, full_path_retriever, save_fig

    quantity_name = "rel_err_n_edges_across_levels" #if sum_model.fc_direction == "cg" else f"rel_err_n_edges_from_{sum_model.top_level}"
    quantity_title = quantity_name[:len("_across_levels")+1] + f"_top_level_{sum_model.top_level}"
    if stripes_level != None:
        quantity_title += f"_stripes_{stripes_level}"
    full_path = sum_model.plots_dir_multi_models + f"/{quantity_title}.pdf"

    if not os.path.exists(full_path):
        from utils import fc_title

        n_edges_across_levels = lambda new_model_name: load_array(full_path_retriever(sum_model, level = None, name = new_model_name, str_dimXBC = "dimX1", meas = quantity_name, stripes_level = stripes_level))

        lw_ampl = 1.2

        # fitn_models = [m for m in models_name if m.startswith("fitn")]
        # local_models = [m for m in models_name if not m.startswith("fitn")]

        for models in model_names:

            # Create subplots
            fig, ax = plt.subplots(figsize=(10, 6))

            scale = 100
            for i, model in enumerate(model_names):

                # set the levels and the rel_error on edges
                levels = np.arange(total_levels) if sum_model.fc_direction == "cg" else np.arange(total_levels - 1, -1, -1)
                edges = n_edges_across_levels(model)

                # find the prefix to be removed from the model name
                fc_label = sum_model.name.split("-")[0]+"-" if "-" in sum_model.name else ""

                # plot the edges
                ax.scatter(levels, edges * scale,
                            marker=markers[i], c = colors[i], s = 50 * lw_ampl, #fc = "none", ,
                            label=model.replace(fc_label, "").replace("DMSM", ""))


            # Customize the plot
            ax.set(xlabel = 'Levels', ylabel = 'Sign.Rel.Err. Number of Edges (%)')
            ax.set_xticks(levels)

            # ax.set_title(f'RelErr of the Number of Edges for {fc_title(sum_model)} models', y = 1.03)
            ax.legend()
            ax.grid(False)

            if sum_model.fc_direction == "fg":
                ax.invert_xaxis()

            # set horizontal grey lines to inspect the (percentage) relative error
            # use [1:-1] in the fc case to avoid shifted major ticks after the plotting of horizontal lines
            # otherwise try: y_ticks = ax.get_yticks().copy(), (after plotting) --> ax.set_yticks(y_ticks)

            for h in ax.get_yticks()[1:-1]:
                ax.hlines(h, 0, total_levels - 1, color='lightgrey', linestyle='--', linewidth=1, zorder=0)


            # Save the plot in multi-models folder
            save_fig(fig, full_path)

            plt.close()

def pagerank_on_internal_nodes(g, gI_dirfunc, intra_size, num_vsplits):
    """
    Plot the Page-Rank for a fixed number of vsplits (num_vsplits): 
    .) x-axis, there would be the full page rank of the intra nodes.
    .) y-axis, the page-rank determined on internal connections
    num_vsplits: integer number of vsplits which are equal to the number of seeds used to select the vI
    """
    
    import math
    from matplotlib import colormaps as cmaps
    from matplotlib.colors import to_hex
    from tqdm import trange
    import os

    intra_size = [intra_size] if isinstance(intra_size, float) else intra_size
    ivec = "_page_rank"
    
    for intra_size in intra_size:

        # check if the folder already exists
        full_path = g.plots_base_dir + f"/PageRank_on_Intra/intra_size{intra_size}/PR_grid_{num_vsplits}.pdf"
        if not os.path.exists(full_path):

            # update the parameters for page rank        
            # p, max_iter, tol, personalize, reverse = kwargs_pr.values()

            # define personlized colors
            colors = cmaps["viridis"](np.linspace(0,1,num_vsplits))

            # Compute grid size (rows, cols) as close to square as possible
            n_cols = math.ceil(math.sqrt(num_vsplits))
            n_rows = math.ceil(num_vsplits / n_cols)

            # define the fig where to store the page-ranks
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False, sharex=True, sharey=True)

            for idx, vsplit in enumerate(trange(num_vsplits, desc="Splitting and Computing the PageRank")):
                row, col = divmod(idx, n_cols) # returns idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # # vsplit nodes, edges in interal
                # vI, eI = g.vsplit_intra_row(v, e, intra_size=intra_size, vsplit=vsplit)
                
                # # update the graph name
                # kwargs_graph.update({'graph_kind': "intra"})
                # gI = sp.graphs.DiGraph(vI, eI, **kwargs_graph)
                
                # # compute the page-rank only in the internal part
                # pr_gI = pagerank_power(gI.adj, p=p, max_iter=max_iter, tol=tol, personalize=personalize, reverse=reverse)
                # gI_path = gI.vars_dir.replace(f"intra_size{gI.intra_size}", f"intra_size{intra_size}").replace(f"vsplit{gI.vsplit}", f"vsplit{vsplit}")
                gI_path = gI_dirfunc(intra_size, vsplit)				
                gI_dict = utils.load_dict(gI_path + "/graph.pkl")
                pr_gI = gI_dict[meas]

                # select the internal node
                idx_IntraNode2Full = list(map(lambda x: g.id_dict.get(x), gI_dict["id_dict"]))
                pr_g_on_I = g.get(ivec)[idx_IntraNode2Full]

                # plot the page rank only on the interal nodes
                _ = ax.plot([pr_g_on_I.min(), pr_g_on_I.max()],
                            [pr_g_on_I.min(), pr_g_on_I.max()],
                            'r--')
                
                ms, alpha = 30, 0.5
                _ = ax.scatter(pr_g_on_I, pr_gI, alpha=alpha, label=f"vsplit {vsplit}", c=to_hex(colors[vsplit]), s=ms)
                _ = ax.set(
                    xlabel='Full-PR on Intra',
                    ylabel='Intra PR',
                    title=f'vsplit {vsplit}'.title(),
                    xscale='log',
                    yscale='log'
                )
                
                # don't plot the grid and legend
                _ = ax.grid(False)
                leg = ax.legend(fontsize=20)
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
                    lh.set_sizes([60])

            # Hide unused subplots
            for idx in range(num_vsplits, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                fig.delaxes(axes[row, col])

            fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

            utils.save_fig(fig, full_path=full_path)
            plt.close()

def _compute_hist2d(x, y, num_bins = 30, axis_scale = "linear"):
    # obtain the 2D density of the pmatrix
    bins = num_bins
    if axis_scale == "log":
        log_bins = lambda a: np.geomspace(start = np.min(a), stop = np.max(a), num = num_bins+1)
        bins = [log_bins(x), log_bins(y)]
    H, xedges, yedges = np.histogram2d(x, y, bins)

    # Histogram does not follow Cartesian convention (see numpy docs), transpose H for visualization purposes.
    H = H.T / x.size

    return H, xedges, yedges

def _plot_hist2d(fig, ax, x, y, num_bins, axis_scale = "log"):

    H, xedges, yedges = _compute_hist2d(x, y, num_bins = num_bins, axis_scale=axis_scale)
    mesh = ax.pcolormesh(xedges, yedges, H, cmap=dep.cmap, norm=axis_scale, zorder=1)
    # im = ax.imshow(H, cmap = dep.cmap, norm = axis_scale, aspect = "auto", 
    #                     origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #                     zorder = 1)

    fig.colorbar(mesh)
    
    # plot also the pearson and spearman correlation coefficients
    from scipy import stats	
    pears_corr = stats.pearsonr(x, y)[0]
    spear_corr = stats.spearmanr(x, y)[0]
    stats = (f'Pears CC = {pears_corr:.3f}\n'
            f'Spear CC = {spear_corr:.3f}')
    bbox = dict(boxstyle='round', fc='whitesmoke', ec='lightgrey', alpha=1)
    ax.text(0.48, 0.87, stats, fontsize=15, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')
    # return im

def ivec_on_internal_nodes(base_dir, g, gI, ivec, num_bins = 100, num_sampled_graphs = 0, ivec_name = "_page_rank"):
    """
    Plot the Page-Rank for a fixed number of vsplits (num_vsplits): 
    .) x-axis, there would be the full page rank of the intra nodes.
    .) y-axis, the page-rank determined on internal connections
    num_vsplits: integer number of vsplits which are equal to the number of seeds used to select the vI
    """
    import os

    full_path = base_dir + f"/{ivec_name.replace('_', '')}_on_intra/num_samples_{int(num_sampled_graphs)}.pdf"

    if True: #not os.path.exists(full_path):
        axis_scale = "log"
        
        # prepare the meas over g and gI
        g_ivec = g.get(ivec_name)
        gI_ivec = gI.get(ivec_name)

        idx_IntraNode2Full = list(map(lambda x: g.id_dict.get(x), gI.id_dict))
        g_ivec_on_I = g_ivec[idx_IntraNode2Full]
        ivec_on_I = ivec[idx_IntraNode2Full]

        fig, axs = plt.subplots(1,2, figsize = (12, 6), sharex=True, sharey=True)

        x = g_ivec_on_I
        _plot_hist2d(fig, axs[0], x, gI_ivec, num_bins = num_bins, axis_scale = axis_scale)
        _plot_hist2d(fig, axs[1], x, ivec_on_I, num_bins = num_bins, axis_scale = axis_scale)

        # plot the identity line, no grid, customize the legend, set the lables and scale
        for i, ax in enumerate(axs):
            
            # plot the reference identity line
            _ = ax.plot([g_ivec_on_I.min(), g_ivec_on_I.max()],
                        [g_ivec_on_I.min(), g_ivec_on_I.max()],
                        'r--', zorder = 1,
                        )

            # set title
            title = "Observed" if i == 0 else "Reconstructed"
            _ = ax.grid(False)

            _ = ax.set(
                        xlabel='Full-PR on Intra',
                        ylabel='Intra PR',
                        xscale=axis_scale,
                        yscale=axis_scale,
                        title = title
                    )

        fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

        utils.save_fig(fig, full_path=full_path)
        plt.close()

def norm_diffs_per_iteration(plots_dir, diff_norms):
    """
    Plot the norm diff against the numb of iteration to inspect convergence properties
    """
    from matplotlib.ticker import MaxNLocator # Import the locator

    fig, ax = plt.subplots(figsize = (12,7))
    axis_scale = 'log'
    msize = 30
    ax.scatter(np.arange(1, len(diff_norms)+1),diff_norms, marker = 'o', color = 'b', s = msize,)
    ax.set(yscale = axis_scale, xlabel = 'Iteration', ylabel = 'Norm Diff')
    # ax.set_xticks(range(len(diff_norms)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_ylim(np.min(diff_norms) * 0.6, None)
    ax.set_axisbelow(True)
    ax.grid(False)

    full_path = plots_dir + "/diff_norms.pdf"
    utils.save_fig(fig, full_path)
    plt.close()

def inset_ivec_vs_rank(ax, x, true_rank, axis_scale = "log", size = 15):
    # in the inset, plot the meas based on the g_ivec_on_I ranking
    inax_w = 0.3
    pos_xy = [0.03, 0.03]
    kwargs_inaxs = {"xscale" : axis_scale, "yscale" : axis_scale}
    inaxs = ax.inset_axes([pos_xy[0], pos_xy[1], inax_w, inax_w], **kwargs_inaxs)
    inaxs.scatter(x, true_rank, marker = 'o', color = dep.obs_color, s = size,)
    inaxs.grid(False)
    inaxs.set(xscale = axis_scale, yscale = axis_scale,)
    
    # force ytick labels, not being present
    from matplotlib.ticker import NullFormatter
    inaxs.yaxis.set_major_formatter(NullFormatter())
    inaxs.yaxis.set_minor_formatter(NullFormatter())
    inaxs.set_xticklabels([])

    return inaxs

def delete_files_with_pattern(base_dir, num_sampled_graphs):
    """
    Walk through all subdirectories from a given directory and delete files
    with a specific pattern in their name.

    Parameters:
    -----------
    base_dir : str
        The base directory to start walking through.
    num_sampled_graphs : int
        The current number of sampled graphs.
    step : int
        The step value to calculate the pattern.

    Returns:
    --------
    None
    """
    import os
    pattern = str(int(num_sampled_graphs))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

def _set_alpha(bars, caps, alpha = 0.5):
    """ 
    From axes, set the alpha to bars and caps of the errorbars
    """
    # set alpha only for bars and caps of the errorbars
    [bar.set_alpha(alpha) for bar in bars]
    [cap.set_alpha(alpha) for cap in caps]

def ivec_on_internal_nodes_vs_rank(base_dir, g, gI, ivec, ivec_std, ivec_name, num_sampled_graphs, num_sigmas = 1):
    """
    Create 2 plots sharing the same x-axis, which is the ranking position (range(1, N))
    Left) meas computed on the sub-internal graph VS ranking;
    Right) Ensemble Average on the frozen edges + reconstructed VS ranking;

    Insets:
    """
    full_path = base_dir + f"/ranked_{ivec_name.replace('_', '')}_on_intra/num_samples_{int(num_sampled_graphs)}.pdf"

    # get the network measures
    g_ivec = g.get(ivec_name)
    gI_ivec = gI.get(ivec_name)

    # select only the intra nodes
    idx_IntraNode2Full = list(map(lambda x: g.id_dict.get(x), gI.id_dict))
    g_ivec_on_I = g_ivec[idx_IntraNode2Full]
    ivec_on_I = ivec[idx_IntraNode2Full]

    # obtain the idx of ranked (descending) meas
    inv_argsort = lambda x: np.argsort(x)[::-1]
    idx_g_ivec_on_I = inv_argsort(g_ivec_on_I)
    idx_gI = inv_argsort(gI_ivec)
    idx_ens_ivec_on_I = inv_argsort(ivec_on_I)

    # plot them
    fig, axs = plt.subplots(1, 2, figsize = (20,7))
    axis_scale, msize = 'log', 15
    title_ivec = ivec_name.lstrip("_").title()
    
    # x-axis will be just increasing values, i.e. ranking position
    x = range(1, len(gI_ivec)+1)

    # scores to assign the ranking
    ranked_g_ivec_on_I = g_ivec_on_I[idx_g_ivec_on_I]

    # === focus on meas obtained by considering only a portion of the network ===
    # plot the measurements as a function of their rankings in a descending order
    axs[0].scatter(x, ranked_g_ivec_on_I, marker = 'o', color = dep.obs_color, s = msize, label = 'Full Network')
    axs[0].scatter(x, gI_ivec[idx_gI], marker = 'x', color = dep.ref_model_color, s = msize, label = 'Internal')
    axs[0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'rank (descending)', ylabel = f'{title_ivec} Values',)
    
    # plot inset measurements as a function of the idx_g_ivec_on_I (here selected nodes are the same)
    inaxs = inset_ivec_vs_rank(axs[0], x, true_rank = ranked_g_ivec_on_I, axis_scale=axis_scale, size = msize)
    inaxs.scatter(x, gI_ivec[idx_g_ivec_on_I], marker = 'x', color = dep.ref_model_color, s = msize)

    # === focus on meas obtained by RECONSTRUCTING the missing parts ===
    # plot the measurements as a function of their rankings in a descending order
    axs[1].scatter(x, ranked_g_ivec_on_I, marker = 'o', color = dep.obs_color, s = msize, label = 'Full Network')
    
    num_sigmas_label = "" if num_sigmas == 1 else num_sigmas
    axs[1].scatter(x, y = ivec_on_I[idx_ens_ivec_on_I], marker = "x", 
                    color = dep.sum_model_color, label = 'Int. + Reconstr.',)
    axs[1].fill_between(x, y1 = ivec_on_I[idx_ens_ivec_on_I] + num_sigmas * ivec_std[idx_ens_ivec_on_I],
                        y2 = ivec_on_I[idx_ens_ivec_on_I] - num_sigmas * ivec_std[idx_ens_ivec_on_I], 
                        color = dep.sum_model_color, capstyle = "butt", label = f'Disp.Int. [-{num_sigmas_label}s, +{num_sigmas_label}s]',
                        alpha = 0.3)
    # _, bars, caps = axs[1].errorbar(x, ivec_on_I[idx_ens_ivec_on_I], yerr = ivec_std[idx_ens_ivec_on_I], 
                    # fmt = 'x', color = dep.sum_model_color, capsize = 5, elinewidth = 1, label = 'Int. + Reconstr.')
    # _set_alpha(bars, caps)
    axs[1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'rank (descending)', ylabel = f'{title_ivec} Values',)
    

    # in the inset, plot the meas based on the g_ivec_on_I ranking
    inaxs = inset_ivec_vs_rank(axs[1], x, true_rank = ranked_g_ivec_on_I, axis_scale=axis_scale, size = msize)
    # _, bars, caps = inaxs.errorbar(x, ivec_on_I[idx_g_ivec_on_I], yerr = ivec_std[idx_g_ivec_on_I], 
    #                 fmt = 'x', color = dep.sum_model_color, capsize = 5, elinewidth = 1,)
    # _set_alpha(bars, caps)
    
    inaxs.scatter(x, y = ivec_on_I[idx_g_ivec_on_I], marker = "x", color = dep.sum_model_color)
    inaxs.fill_between(x, y1 = ivec_on_I[idx_g_ivec_on_I] + num_sigmas * ivec_std[idx_g_ivec_on_I],
                        y2 = ivec_on_I[idx_g_ivec_on_I] - num_sigmas * ivec_std[idx_g_ivec_on_I], 
                        color = dep.sum_model_color, capstyle = "butt", alpha = 0.3)

    for ax in axs:
        ax.legend(markerscale=2)
        ax.set_axisbelow(True)
        ax.grid(True)

    fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    
    utils.save_fig(fig, full_path=full_path)
    plt.close()

def plot_local_fonts(corpkey):
    import matplotlib.pyplot as plt

    from matplotlib import font_manager
    import os

    dir_ = "outputs"
    not_exists = True if corpkey else not os.path.exists(dir_ + "/fonts.pdf")
    if  not_exists:

        os.makedirs(dir_, exist_ok = True)

        # Get all available fonts
        available_fonts = sorted(list(set(f.name for f in font_manager.fontManager.ttflist)))

        # Create a figure with subplots for each font
        fig, axs = plt.subplots(len(available_fonts) // 5 + 1, 5, figsize=(20, len(available_fonts) * 0.3))
        axs = axs.flatten()

        # Plot each font
        for i, font in enumerate(available_fonts):
            axs[i].text(0.5, 0.5, font, ha='center', va='center', fontsize=10, fontfamily=font)
            axs[i].set_title(font, fontsize=8)
            axs[i].axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # Adjust layout
        plt.tight_layout()

        utils.save_fig(fig, dir_ + "/fonts.pdf")

        plt.close()
        