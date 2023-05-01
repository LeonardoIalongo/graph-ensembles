import numpy as np
import pandas as pd
import sys

sys.path.insert(0,'/Users/sylvainbangma/Documents/Github/graph-ensembles')
import src.graph_ensembles as ge 

v = pd.DataFrame([['ING', 'NL'],
                 ['ABN', 'NL'],
                 ['BNP', 'FR'],
                 ['BNP', 'IT']],
                 columns=['name', 'country'])

e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6],
                 ['BNP', 'FR', 'ABN', 'NL', 2.3e7],
                 ['BNP', 'IT', 'ABN', 'NL', 7e5 + 3e3],
                 ['ABN', 'NL', 'BNP', 'FR', 1e4],
                 ['ABN', 'NL', 'ING', 'NL', 4e5]],
                 columns=['creditor', 'c_country',
                          'debtor', 'd_country',
                          'value'])

g = ge.Graph(v, e, v_id=['name', 'country'],
             src=['creditor', 'c_country'],
             dst=['debtor', 'd_country'],
             weight='value')


model = ge.ScaleInvariantModel_selfloops(g)

fitted_model = model.fit()

param = fitted_model.param
print(param)