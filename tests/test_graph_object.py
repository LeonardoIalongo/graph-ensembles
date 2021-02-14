""" Test graph class object creation and attributes. """
import graph_ensembles as ge
import pandas as pd
import pytest


class TestSimpleGraph():
    v = pd.DataFrame([['ING', 'NL', 1e12],
                     ['ABN', 'NL', 5e11],
                     ['BNP', 'FR', 13e12]],
                     columns=['name', 'country', 'assets'])

    e = pd.DataFrame([['ING', 'ABN', 1e6, 'interbank'],
                     ['BNP', 'ABN', 2.3e7, 'external'],
                     ['BNP', 'ABN', 1.7e5, 'interbank'],
                     ['ABN', 'BNP', 1e4, 'interbank'],
                     ['ABN', 'ING', 4e5, 'external']],
                     columns=['creditor', 'debtor', 'value', 'type'])

    def test_wrong_input(self):
        with pytest.raises(Exception) as e_info:
            ge.Graph([[1], [2], [3]], [[1, 2], [3, 1]])

        msg = 'Only dataframe input supported.'
        assert e_info.value.args[0] == msg

    def test_instanciation_names(self):
        g = ge.Graph(self.v, self.e, id_col='name', src_col='creditor',
                     dst_col='debtor')

        assert isinstance(g, ge.Graph)
