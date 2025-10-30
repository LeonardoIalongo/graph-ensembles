import matplotlib as mpl
import matplotlib.pyplot as plt

# set plot parameters
mpl.rc("xtick", direction = "in")
mpl.rc("ytick", direction = "in")
mpl.rc('xtick.minor',size= 4, )
mpl.rc('ytick.minor',size= 4, )
mpl.rc('xtick.major',size= 8, )
mpl.rc('ytick.major',size= 8, )
mpl.rcParams.update({"axes.grid" : True})
mpl.rcParams['axes.formatter.min_exponent'] = 1
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['figure.subplot.hspace'] = 0.25
plt.rcParams['figure.subplot.wspace'] = 0.25

obs_color = "#003049"
obs_edgecolor = "#001C2A"
obs_ms = 60
obs_marker = "^"
ref_model_color ="#57A773" #"#157145" #  # "#157145" #"#8FC93A" #"#f77f00" #"#009DDC" #"#0714b1" #"darkviolet"
ref_model_edgecolor = "#15291C" #"#FCBF49" # #"#FCBF49" #"#E3B505"s
ref_model_ms = 7
ref_model_marker = "o"
sum_model_color = "#d62828" #"#FCBF49" # #"#FCBF49" #"#E3B505"s
sum_model_edgecolor = "#470d0d" #"#FCBF49" # #"#FCBF49" #"#E3B505"s
sum_model_ms = 7
sum_model_marker = "H"
azure_color = "#22AED1"
azure_edgecolor = "#0A2F38"

# custom cmap
# cmap = mpl.colors.LinearSegmentedColormap.from_list("", [obs_color,sum_model_color,ref_model_color, "#5FAD56", "#F2C14E"])
cmap_colors = [obs_color,sum_model_color,"#fb8b24", "#fcbf49", "#eae2b7"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", cmap_colors)

