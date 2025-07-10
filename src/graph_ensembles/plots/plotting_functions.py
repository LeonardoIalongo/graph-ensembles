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

def pagerank_on_internal_nodes(g, gI, intra_size, num_splits):
	"""
	Plot the Page-Rank for a fixed number of splits (num_splits): 
	.) x-axis, there would be the full page rank of the intra nodes.
	.) y-axis, the page-rank determined on internal connections
	num_splits: integer number of splits which are equal to the number of seeds used to select the vI
	"""
	
	import math
	from matplotlib import colormaps as cmaps
	from matplotlib.colors import to_hex
	from tqdm import trange
	import os
	from fast_pagerank import pagerank_power

	intra_size = [intra_size] if isinstance(intra_size, float) else intra_size
	net_meas = "_page_rank"
	
	for intra_size in intra_size:

		# check if the folder already exists
		full_path = g.plots_base_dir + f"/PageRanks_on_Intra/intra_size{intra_size}/PR_grid_{num_splits}.pdf"
		if not os.path.exists(full_path):

			# update the parameters for page rank        
			# p, max_iter, tol, personalize, reverse = kwargs_pr.values()

			# define personlized colors
			colors = cmaps["viridis"](np.linspace(0,1,num_splits))

			# Compute grid size (rows, cols) as close to square as possible
			n_cols = math.ceil(math.sqrt(num_splits))
			n_rows = math.ceil(num_splits / n_cols)

			# define the fig where to store the page-ranks
			fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False, sharex=True, sharey=True)

			for idx, split in enumerate(trange(num_splits, desc="Splitting and Computing the PageRank")):
				row, col = divmod(idx, n_cols) # returns idx // n_cols, idx % n_cols
				ax = axes[row, col]

				# # split nodes, edges in interal
				# vI, eI = g.split_intra_row(v, e, intra_size=intra_size, split=split)
				
				# # update the graph name
				# kwargs_graph.update({'graph_kind': "intra"})
				# gI = sp.graphs.DiGraph(vI, eI, **kwargs_graph)
				
				# # compute the page-rank only in the internal part
				# pr_gI = pagerank_power(gI.adj, p=p, max_iter=max_iter, tol=tol, personalize=personalize, reverse=reverse)
				gI_path = gI.vars_dir.replace(f"intra_size{gI.intra_size}", f"intra_size{intra_size}").replace(f"vert_split{gI.vert_split}", f"vert_split{split}")
				
				gI_dict = utils.load_dict(gI_path + "/graph.pkl")
				pr_gI = gI_dict[net_meas]

				# select the internal node
				idx_IntraNode2Full = list(map(lambda x: g.id_dict.get(x), gI_dict["id_dict"]))
				pr_g_on_I = g.get(net_meas)[idx_IntraNode2Full]

				# plot the page rank only on the interal nodes
				_ = ax.plot([pr_g_on_I.min(), pr_g_on_I.max()],
							[pr_g_on_I.min(), pr_g_on_I.max()],
							'r--')
				
				ms, alpha = 30, 0.5
				_ = ax.scatter(pr_g_on_I, pr_gI, alpha=alpha, label=f"Split {split}", c=to_hex(colors[split]), s=ms)
				_ = ax.set(
					xlabel='Full-PR on Intra',
					ylabel='Intra PR',
					title=f'split {split}'.title(),
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
			for idx in range(num_splits, n_rows * n_cols):
				row, col = divmod(idx, n_cols)
				fig.delaxes(axes[row, col])

			fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

			utils.save_fig(fig, full_path=full_path)
			plt.close()


def internal_net_meas_obs_vs_reconstr(model, intra_size, split, g, gI, ens_mean_net_meas, num_sampled_graphs):
	"""
	Plot the Page-Rank for a fixed number of splits (num_splits): 
	.) x-axis, there would be the full page rank of the intra nodes.
	.) y-axis, the page-rank determined on internal connections
	num_splits: integer number of splits which are equal to the number of seeds used to select the vI
	"""
	
	from matplotlib import colormaps as cmaps
	from matplotlib.colors import to_hex
	import os

	net_meas = "_page_rank"
	
	# check if the folder already exists
	full_path = g.plots_base_dir + f"/PageRanks_on_Intra/intra_size{intra_size}/Reconstructed/{model.fit_method}/split{split}_{num_sampled_graphs}.pdf"
	if not os.path.exists(full_path):
		
		# prepare the net_meas over g and gI
		g_net_meas = g.get(net_meas)
		gI_net_meas = gI.get(net_meas)
		
		idx_IntraNode2Full = list(map(lambda x: g.id_dict.get(x), gI.id_dict))
		g_net_meas_on_I = g_net_meas[idx_IntraNode2Full]
		ens_mean_net_meas_on_I = ens_mean_net_meas[idx_IntraNode2Full]

		# define the fig where to store the page-ranks
		fig, axs = plt.subplots(1,2, figsize = (12, 6), sharex=True, sharey=True)

		ms, alpha = 30, .5

		# plot the observed gI_net_meas
		_ = axs[0].scatter(g_net_meas_on_I, gI_net_meas, alpha=alpha, c=dep.obs_color, s=ms, zorder = 1)
		axs[0].set_title("Observed")
		_ = axs[1].scatter(g_net_meas_on_I, ens_mean_net_meas_on_I, alpha=alpha, c=dep.ref_model_color, s=ms, zorder = 1)
		axs[1].set_title("Reconstructed")
		
		# plot the identity line, no grid, customize the legend, set the lables and scale
		for ax in axs:
			_ = ax.plot([g_net_meas_on_I.min(), g_net_meas_on_I.max()],
						[g_net_meas_on_I.min(), g_net_meas_on_I.max()],
						'r--', zorder = 0
						)

			_ = ax.grid(False)

			_ = ax.set(
						xlabel='Full-PR on Intra',
						ylabel='Intra PR',
						xscale='log',
						yscale='log'
					)

		fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

		utils.save_fig(fig, full_path=full_path)
		plt.close()