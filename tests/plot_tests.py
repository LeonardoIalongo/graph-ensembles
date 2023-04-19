import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame([['1',  0.76],
                 ['50', 0.8],
                 ['100', 0.93],
                 ['150', 0.97],
                 ['200', 1],
                 ['250', 1.05],
                 ['300', 1.13],
                 ['350',1.21],
                 ['400',1.37]],
                 columns=['aggr_level', 'weighted_param'])
sns.set_style("darkgrid")
sns.scatterplot(data=data, x='aggr_level',y='weighted_param').set(title="Parameter as a function of aggregation level")
plt.show()