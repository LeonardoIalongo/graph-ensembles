from graph_ensembles.dependencies import *
from .. import dependencies as dep
from graph_ensembles.utils import load_meas, save_fig
from graph_ensembles import sparse as sp
import numpy as np
from .. import utils


def ccdf_deg_out_in(g, gI, model):
    full_path = model.plots_dir + "/ccdf_deg_out_in.png"

    def _ccdf_vs_deg(deg):
        def normalized_ccdf(arr):
            ccdf = np.cumsum(arr)[::-1]
            # Problems with ccdf/ccdf[0]. Therefore, use np.divide
            ccdf = np.divide(ccdf, ccdf[0])
            return ccdf

        return np.sort(deg), normalized_ccdf(deg)

    def plot_ccdf(axs, deg_out, deg_in, color, lw = 5, label = "ciao"):
        x, y = _ccdf_vs_deg(deg_out)
        axs[0].step(x, y, color = color, lw = lw, label = label)
        x, y = _ccdf_vs_deg(deg_in)
        axs[1].step(x, y, color = color, lw = lw, label = label)

    fig, axs = plt.subplots(1, 2, figsize = (20,7))

    plot_ccdf(axs, g.out_degree(), g.in_degree(), lw = 9, color = dep.obs_color, label = "Full Network")
    # plot_ccdf(axs, gI.out_degree(), gI.in_degree(), lw = 7, color = dep.ref_model_color, label = "Internal")
    plot_ccdf(axs, model.expected_out_degree(),  model.expected_in_degree(), lw = 5, color = dep.sum_model_color, label = f'Rec. w/ {model.fit_method_title}')

    axis_scale = "log"
    for i, ax in enumerate(axs):
        out_in_label = "Out" if i == 0 else "In"
        ax.set(xlabel = f'{out_in_label}-Degrees', ylabel = 'CCDF', xscale = axis_scale, yscale = "linear")
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid(True)

    fig.tight_layout()

    utils.save_fig(fig, full_path=full_path)
    plt.close()

def annd_vs_deg(g, gI, model, measures):
    for a in [i for i in measures if "annd" in i]:
        ddir, ndir = a.split("_")[2:]
        annd_vs_deg_out_in(g, gI, model, ddir, ndir)

def annd_vs_deg_out_in(g, gI, model, ddir = "out", ndir = "in"):

    num_sigmas = model.num_sigmas
    num_sigmas_label = "" if num_sigmas == 1 else num_sigmas

    meas = f"_annd_{ddir}_{ndir}"
    deg_meas = f"_{ddir}_degree"
    full_path = model.plots_dir + f"/deg_annd_cc/{meas[1:]}.png"

    fig, axs = plt.subplots(figsize = (20,7))
    axis_scale = 'log' 
    obs_s, exp_s = 30, 30
    alpha = 0.4
    bar_alpha = 0.2

    # import the degree (x-axis) and the annd (y-axis)
    deg_x, deg_y, deg_z, deg_z_std = g.__dict__[deg_meas], gI.__dict__[deg_meas], model.__dict__[deg_meas], model.__dict__[deg_meas+"_std"] 
    x, y, z, z_std = g.__dict__[meas], gI.__dict__[meas], model.__dict__[meas], model.__dict__[meas+"_std"] 

    # filter deg and annd with respect to the internal nodes
    proj_on_I = lambda x: x[gI.internal_nodes]
    deg_x, deg_z, deg_z_std = proj_on_I(deg_x), proj_on_I(deg_z), proj_on_I(deg_z_std)
    x, z, z_std = proj_on_I(x), proj_on_I(z), proj_on_I(z_std)

    _, bars, caps = axs.errorbar(
        x = deg_z, xerr = num_sigmas * deg_z_std, y = z, yerr = num_sigmas * z_std, fmt=dep.sum_model_marker, color=dep.sum_model_color,
        label=f'Rec. w/ {model.fit_method_title} +- {num_sigmas_label}s', capsize=5, alpha = alpha, ms = np.sqrt(exp_s), mec = "k")
    _set_alpha(bars, caps, alpha = bar_alpha)
    
    # axs.scatter(deg_z, z, marker = "x", color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')
    axs.scatter(deg_x, x, marker = dep.obs_marker, color = dep.obs_color, s = obs_s, label = 'Full Network', alpha = alpha)
    axs.scatter(deg_y, y, marker = dep.ref_model_marker, color = dep.ref_model_color, s = exp_s, label = 'Internal', alpha = alpha, ec = "k")

    ax = axs
    ax.set(xscale = axis_scale, yscale = axis_scale,)
    ax.set(xlabel = f'{ddir.title()}-Degrees', ylabel = f'Avg.Ne.Ne.Deg. {ddir.title()}-{ndir.title()}',)
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(markerscale = 2)

    fig.tight_layout()

    utils.save_fig(fig, full_path=full_path)
    plt.close()

def exp_deg_out_in(g, gI, model):
    """Plot Internal Out and In Degrees as computed in the Full Network, Internal or Int+Reconstructed Model."""
    full_path = model.plots_dir + "/deg_annd_cc/deg_out_in.png"

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    axis_scale = 'log'
    obs_s, exp_s = 30, 30
    alpha = 0.4
    bar_alpha = 0.2
    
    num_sigmas = model.num_sigmas
    num_sigmas_label = "" if num_sigmas == 1 else num_sigmas

    # import the out-degree with error bars
    x, y, mu, sigma = g._out_degree, gI._out_degree, model._out_degree, model._out_degree_std

    # project the ground truth and expected values on I
    proj_on_I = lambda x: x[gI.internal_nodes]
    x, mu, sigma = proj_on_I(x), proj_on_I(mu), proj_on_I(sigma)

    _, bars, caps = axs[0].errorbar(
        x, mu, yerr=num_sigmas * sigma, fmt=dep.sum_model_marker, color=dep.sum_model_color,
        label=f'Rec. w/ {model.fit_method_title} +- {num_sigmas_label}s', capsize=5, ms = np.sqrt(exp_s), mec = "k", alpha = alpha
    )
    _set_alpha(bars, caps, alpha = bar_alpha)

    axs[0].scatter(x, x, marker=dep.obs_marker, color=dep.obs_color, s=obs_s, label='Full Network')
    axs[0].scatter(x, y, marker=dep.ref_model_marker, color=dep.ref_model_color, s=exp_s, label='Internal', ec = "k", alpha = alpha)

    # Plot the in-degree with error bars
    x, y, mu, sigma = g._in_degree, gI._in_degree, model._in_degree, model._in_degree_std
    x, mu, sigma = proj_on_I(x), proj_on_I(mu), proj_on_I(sigma)

    _, bars, caps = axs[1].errorbar(
        x, mu, yerr=num_sigmas * sigma, fmt=dep.sum_model_marker, color=dep.sum_model_color,
        label=f'Rec. w/ {model.fit_method_title} +- {num_sigmas_label}s', capsize=5,
        ms = np.sqrt(exp_s), mec = "k", alpha = alpha)
    _set_alpha(bars, caps, alpha = bar_alpha)

    axs[1].scatter(x, x, marker=dep.obs_marker, color=dep.obs_color, s=obs_s, label='Full Network')
    axs[1].scatter(x, y, marker=dep.ref_model_marker, color=dep.ref_model_color, s=exp_s, label='Internal', ec = "k", alpha = alpha)

    for i, ax in enumerate(axs):
        ax.set(xscale=axis_scale, yscale=axis_scale)
        out_in_label = "Out" if i == 0 else "In"
        ax.set(xlabel=f'{out_in_label}-Degrees', ylabel=f'{out_in_label}-Degrees')
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.legend(markerscale=2)

    fig.tight_layout()

    utils.save_fig(fig, full_path=full_path)
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
    # pears_corr = stats.pearsonr(x, y)[0]
    spear_corr = stats.spearmanr(x, y)[0]
    # stats = (f'Pears CC = {pears_corr:.3f}\n'
    stats = (f'Spear CC = {spear_corr:.3f}')
    bbox = dict(boxstyle='round', fc='whitesmoke', ec='lightgrey', alpha=1)
    ax.text(0.54, 0.93, stats, fontsize=15, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')
    # return im

def pr_on_internal_nodes(model, g, gI, num_bins = 100):
    """
    Plot the Page-Rank for a fixed number of vsplits (num_vsplits): 
    .) x-axis, there would be the full page rank of the intra nodes.
    .) y-axis, the page-rank determined on internal connections
    num_vsplits: integer number of vsplits which are equal to the number of seeds used to select the vI
    """
    old_font = mpl.rcParams['font.size']

    mpl.rcParams["font.size"] = 18
    import os
    full_path = model.plots_dir + f"/{g._pr_name}_on_intra.png"

    if True: #not os.path.exists(full_path):
        axis_scale = "log"
        
        fig, axs = plt.subplots(1,2, figsize = (12, 6), sharex=True, sharey=True)

        x = g._pr_on_I
        _plot_hist2d(fig, axs[0], x, gI._pr, num_bins = num_bins, axis_scale = axis_scale)
        _plot_hist2d(fig, axs[1], x, model._pr_on_I, num_bins = num_bins, axis_scale = axis_scale)

        # plot the identity line, no grid, customize the legend, set the lables and scale
        for i, ax in enumerate(axs):
            
            # plot the reference identity line
            _ = ax.plot([x.min(), x.max()],
                        [x.min(), x.max()],
                        'r--', zorder = 1,
                        )

            # set title
            title = "Observed" if i == 0 else f"Rec. w/ {model.fit_method_title}"
            _ = ax.grid(False)

            _ = ax.set(
                        xlabel='Full-PR on Internal Nodes',
                        ylabel='Intra PR',
                        xscale=axis_scale,
                        yscale=axis_scale,
                        title = title
                    )

        fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

        utils.save_fig(fig, full_path=full_path)
        plt.close()
    mpl.rcParams["font.size"] = old_font

def out_in_degree_internal_VS_restricted(g, gI, model):

    full_path = model.plots_dir + f"/out_in_degree_internal_vs_restricted.png"
    
    fig, axs = plt.subplots(1, 2, figsize = (24,8))
    axis_scale = 'log'
    obs_s, exp_s = 100, 50

    # out direction
    x, y = g._out_degree_on_I, gI._out_degree
    _ = axs[0].plot([x.min(), x.max()], [x.min(), x.max()], ls = '--', color = "grey", zorder = -1,)
    axs[0].scatter(x, y, marker = dep.ref_model_marker, color = dep.ref_model_color, s = exp_s * 1.4, label = 'Internal Out-Degree')

    y = model._out_degree_on_I
    axs[0].scatter(x, y, marker = dep.sum_model_marker, color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')
    # plot the reference identity line


    # in direction
    x, y = g._in_degree_on_I, gI._in_degree
    _ = axs[1].plot([x.min(), x.max()], [x.min(), x.max()], ls = '--', color = "grey", zorder = -1,)
    axs[1].scatter(x, y, marker = dep.ref_model_marker, color = dep.ref_model_color, s = exp_s * 1.4, label = 'Internal In-Degree')
    
    y = model._in_degree_on_I
    axs[1].scatter(x, y, marker = dep.sum_model_marker, color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')


    axis_scale = "log" #if out_min == 0 or in_min == 0 else "log"
    for i, ax in enumerate(axs):
        lgd = ax.legend()
        for legend_handle in lgd.legend_handles:
            legend_handle.set_sizes([100])
        
        xy_label = "Out-" if i == 0 else "In-"
        ax.set(xscale = axis_scale, yscale = axis_scale, xlabel = xy_label + 'Degree', ylabel = "Estimated " + xy_label + 'Degree',)
        ax.set_axisbelow(True)
        ax.grid(True)
    
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

    full_path = plots_dir + "/diff_norms.png"
    utils.save_fig(fig, full_path)
    plt.close()

def inset_pr_vs_rank(ax, x, true_rank, axis_scale = "log", size = 15, zorder = 1):
    # in the inset, plot the meas based on the g_pr_on_I ranking
    inax_w = 0.3
    pos_xy = [0.03, 0.03]
    kwargs_inaxs = {"xscale" : axis_scale, "yscale" : axis_scale}
    inaxs = ax.inset_axes([pos_xy[0], pos_xy[1], inax_w, inax_w], **kwargs_inaxs)
    inaxs.scatter(x, true_rank, marker = 'o', color = dep.obs_color, s = size,zorder = zorder)
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

def pr_on_internal_nodes_vs_rank(model, g, gI):
    """
    Create 2 plots sharing the same x-axis, which is the ranking position (range(1, N))
    Left) meas computed on the sub-internal graph VS ranking;
    Right) Ensemble Average on the frozen edges + reconstructed VS ranking;

    Insets:
    """
    full_path = model.plots_dir + f"/ranked_{g._pr_name}_on_intra.png"

    # plot them
    fig, axs = plt.subplots(1, 2, figsize = (20,7))
    axis_scale, msize = 'log', 15
    inset_alpha = 0.3
    inset_zorder_exp = 0
    num_sigmas = model.num_sigmas
    num_sigmas_label = "" if num_sigmas == 1 else num_sigmas
    title_pr = g._pr_name.title()
    
    # x-axis will be just increasing values, i.e. ranking position
    x = range(1, len(gI._pr)+1)

    # scores to assign the ranking
    g_pr_on_I_desc = g._pr_on_I_desc
    g_pr_on_I_rank = g._pr_on_I_rank
    
    # === focus on meas obtained by considering only a portion of the network ===
    # plot the measurements as a function of their rankings in a descending order
    axs[0].scatter(x, gI._pr_desc, marker = 'x', color = dep.ref_model_color, s = msize, label = 'Internal')
    axs[0].scatter(x, g_pr_on_I_desc, marker = 'o', color = dep.obs_color, s = msize, label = 'Full Network')
    axs[0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Rank (descending)', ylabel = f'{title_pr} Values',)
    
    # plot the reorder gI._pr with respect to the full-network ranking, i.e. idx_g_pr_on_I
    inaxs = inset_pr_vs_rank(axs[0], x, true_rank = g_pr_on_I_desc, axis_scale=axis_scale, size = msize, zorder = 1)
    inaxs.scatter(x, gI._pr[g_pr_on_I_rank], marker = 'x', color = dep.ref_model_color, s = msize, zorder = inset_zorder_exp)

    # === focus on meas obtained by RECONSTRUCTING the missing parts ===
    # plot the reorder model._pr with respect to the full-network ranking, i.e. idx_g_pr_on_I
    mu, sigma = model._pr_on_I_desc, model._pr_on_I_std_desc
    axs[1].scatter(x, y = mu, marker = "x", 
                    color = dep.sum_model_color, label = f'Rec. w/ {model.fit_method_title}',)
    axs[1].fill_between(x, y1 = mu + num_sigmas * sigma,
                        y2 = mu - num_sigmas * sigma, 
                        color = dep.sum_model_color, label = f'Disp.Int. [-{num_sigmas_label}s, +{num_sigmas_label}s]',
                        alpha = inset_alpha)
    axs[1].scatter(x, g_pr_on_I_desc, marker = 'o', color = dep.obs_color, s = msize, label = 'Full Network')
    axs[1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Rank (descending)', ylabel = f'{title_pr} Values',)
    

    # in the inset, plot the meas based on the g_pr_on_I ranking
    inaxs = inset_pr_vs_rank(axs[1], x, true_rank = g_pr_on_I_desc, axis_scale=axis_scale, size = msize, zorder = 1)

    # create mu, std arrays and plot scatter + fill between curves
    mu, sigma = model._pr_on_I[g_pr_on_I_rank], model._pr_std_on_I[g_pr_on_I_rank]
    inaxs.scatter(x, y = mu, marker = "x", color = dep.sum_model_color, zorder = inset_zorder_exp)
    inaxs.fill_between(x, y1 = mu + num_sigmas * sigma, y2 = mu - num_sigmas * sigma, 
                        color = dep.sum_model_color, alpha = inset_alpha, zorder = inset_zorder_exp)

    for ax in axs:
        ax.legend(markerscale=2)
        ax.set_axisbelow(True)
        ax.grid(True)

    fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    
    utils.save_fig(fig, full_path=full_path)
    plt.close()

def topN_overlap_pr(gI, model):
    """
    Over the N-firms with highest pr values, plot the average overlap and the total relative error over all the sampled networks
    """
    
    full_path = model.plots_dir + f"/topN_overlap_pr_{gI._pr_name}.png"
    num_sigmas = model.num_sigmas
    num_sigmas_label = "" if num_sigmas == 1 else num_sigmas

    fig, ax = plt.subplots(figsize = (12,7))
    axis_scale = 'log'
    obs_s, exp_s = 60, 60

    # internal topN overlap
    ax.scatter(gI._topN_overlap_pr_range, gI._topN_overlap_pr, marker = 'o', color = dep.ref_model_color, s = exp_s * 1.4, label = 'Internal')
    
    # expected topN overlap on average
    # _, bars, caps = ax.errorbar(
    #     x = model._topN_overlap_pr_range, y = model._topN_overlap_pr, yerr = num_sigmas * model._topN_overlap_pr_std, fmt=dep.sum_model_marker, color=dep.sum_model_color,
    #     label=f'Rec. w/ {model.fit_method_title} +- {num_sigmas_label}s', capsize=5,
    # )
    # _set_alpha(bars, caps, alpha = 0.5)

    # expected topN overlap over mean "#48ACF0" #4E937A
    ax.scatter(model._topN_overlap_pr_on_I_range, model._topN_overlap_pr_on_I, marker = dep.sum_model_marker, color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')
    
    ax.set(xscale = axis_scale, yscale = "linear", xlabel = 'TopN (descending)', ylabel = 'Overlap (%)',)
    
    ax.legend()
    ax.grid(False)

    fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    utils.save_fig(fig, full_path=full_path)
    plt.close()

def topN_overlap_pr_deg_stre(g, gI, model):

    full_path = model.plots_dir + f"/topN_overlap_pr_deg_stre.png"
    
    fig, axs = plt.subplots(1, 2, figsize = (24,8))
    axis_scale = 'log'
    obs_s, exp_s = 100, 50

    # out direction
    x, y = g._topN_overlap_out_degree_on_I_range, g._topN_overlap_out_degree_on_I
    axs[0].scatter(x, y, marker = dep.obs_marker, color = dep.obs_color, s = obs_s * 1.4, label = 'Out-Degree')

    x, y = g._topN_overlap_out_strength_on_I_range, g._topN_overlap_out_strength_on_I
    axs[0].scatter(x, y, marker = dep.obs_marker, color = dep.azure_color, s = obs_s, label = 'Out-Strength')

    x, y = gI._topN_overlap_out_degree_range, gI._topN_overlap_out_degree
    axs[0].scatter(x, y, marker = dep.ref_model_marker, color = dep.ref_model_color, s = exp_s * 1.4, label = 'Internal Out-Degree')

    x, y = model._topN_overlap_out_degree_on_I_range, model._topN_overlap_out_degree_on_I
    axs[0].scatter(x, y, marker = dep.sum_model_marker, color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')


    # in direction
    x, y = g._topN_overlap_in_degree_on_I_range, g._topN_overlap_in_degree_on_I
    axs[1].scatter(x, y, marker = dep.obs_marker, color = dep.obs_color, s = obs_s * 1.4, label = 'In-Degree')

    x, y = g._topN_overlap_in_strength_on_I_range, g._topN_overlap_in_strength_on_I
    axs[1].scatter(x, y, marker = dep.obs_marker, color = dep.azure_color, s = obs_s, label = 'In-Strength')

    x, y = gI._topN_overlap_in_degree_range, gI._topN_overlap_in_degree
    axs[1].scatter(x, y, marker = dep.ref_model_marker, color = dep.ref_model_color, s = exp_s * 1.4, label = 'Internal In-Degree')

    x, y = model._topN_overlap_in_degree_on_I_range, model._topN_overlap_in_degree_on_I
    axs[1].scatter(x, y, marker = dep.sum_model_marker, color = dep.sum_model_color, s = exp_s, label = f'Rec. w/ {model.fit_method_title}')

    out_min = np.min([g._topN_overlap_out_degree_on_I[0], gI._topN_overlap_out_degree[0], g._topN_overlap_out_strength_on_I[0], model._topN_overlap_out_degree_on_I[0]])
    in_min = np.min([g._topN_overlap_in_degree_on_I[0], gI._topN_overlap_in_degree[0], g._topN_overlap_in_strength_on_I[0], model._topN_overlap_in_degree_on_I[0]])
    
    yscale = "linear" if out_min == 0 or in_min == 0 else "log"
    for _, ax in enumerate(axs):
        lgd = ax.legend()
        for legend_handle in lgd.legend_handles:
            legend_handle.set_sizes([100])
        
        ax.set(ylim = [None, 1.05], xscale = "log", yscale = yscale, xlabel = 'TopN (descending)', ylabel = 'Overlap',)
        ax.set_axisbelow(True)
        ax.grid(True)
    
    fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    utils.save_fig(fig, full_path=full_path)
    plt.close()

def topN_overlap_pr_tot_rel_err_on_avg_pr(g, gI, model):
    """Over the N-firms with highest pr values, plot the overlap their overal and the total relative error"""
    
    full_path = model.plots_dir + f"/topN_{g._pr_name}_on_intra_not_ensemble.png"
    intervals = gI._intervals

    if True:
        topN_arr = lambda v: [v[:i] for i in intervals]

        # observed and expected by the model

        # 1. Calculate overlap between g_topN_pr_on_I_rank and model_topN_pr_on_I_rank
        overlap_perc = lambda r: np.array([np.intersect1d(g_topN, exp_topN).size / g_topN.size for g_topN, exp_topN in zip(g._topN_pr_on_I_rank, r)])
        topN_rel_err = lambda r: np.array([utils.tot_rel_err(exp_topN, g_topN) * 100 for g_topN, exp_topN in zip(g._topN_pr, r)])

        # obtain the slicing of the page-rank ranking with respect to intervals
        model_topN_pr_on_I_rank = topN_arr(model._pr_on_I_rank)

        # calculate the overlap
        g_model_overlap = overlap_perc(model_topN_pr_on_I_rank)

        # obtain the slicing of the relative error ranking with respect to intervals
        model_topN_pr = topN_arr(model._pr_on_I)
        g_model_rel_err = topN_rel_err(model_topN_pr)

        fig, axs = plt.subplots(1, 2, figsize = (20,7))
        axis_scale = 'log'
        obs_s, exp_s = 60, 60
        axs[0].scatter(intervals, gI._topN_overlap_pr, marker = 'o', color = dep.ref_model_color, s = exp_s, label = 'Internal')
        axs[0].scatter(intervals, g_model_overlap, marker = dep.sum_model_marker, color = dep.sum_model_color, s = obs_s, label = f'Rec. w/ {model.fit_method_title}')
        axs[0].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Top N Firms', ylabel = 'Overlap (%)',)

        axs[1].scatter(intervals, gI._topN_tot_rel_err,  marker = 'o', color = dep.ref_model_color, s = exp_s, label = 'Internal')
        axs[1].scatter(intervals, g_model_rel_err, marker = dep.sum_model_marker, color = dep.sum_model_color, s = obs_s, label = f'Rec. w/ {model.fit_method_title}')
        axs[1].set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Top N Firms', ylabel = 'Total Relative Error (%)',)

        for ax in axs:
            ax.legend()
            ax.grid(False)

        fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

        utils.save_fig(fig, full_path=full_path)
        plt.close()

def plot_local_fonts(corpkey):
    import matplotlib.pyplot as plt

    from matplotlib import font_manager
    import os

    dir_ = "outputs"
    not_exists = True if corpkey else not os.path.exists(dir_ + "/fonts.png")
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

        utils.save_fig(fig, dir_ + "/fonts.png")

        plt.close()