""" Test graph class object creation and attributes. """
import graph_ensembles as ge
import pandas as pd
import numpy as np
import pytest


class TestDirectedGraph():
    v = pd.DataFrame([['ING'], ['ABN'], ['BNP']],
                     columns=['name'])

    e = pd.DataFrame([['ING', 'ABN'],
                     ['BNP', 'ABN'],
                     ['ABN', 'BNP'],
                     ['BNP', 'ING']],
                     columns=['creditor', 'debtor'])

    _e = np.sort(np.rec.array([(0, 1), (2, 1), (1, 2), (2, 0)],
                              dtype=[('src', np.uint8), ('dst', np.uint8)]))

    def test_instanciation_names(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.DirectedGraph)
        assert np.all(g.e == self._e), g.e

    def test_duplicated_vertices(self):
        v = pd.DataFrame([['ING'], ['ABN'], ['BNP'], ['ABN']],
                         columns=['name'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(v, self.e, v_id='name', src='creditor', dst='debtor')

            msg = 'There is at least one repeated id in the vertex dataframe.'
            assert e_info.value.args[0] == msg

    def test_duplicated_edges(self):
        e = pd.DataFrame([['ING', 'ABN'],
                         ['BNP', 'ABN'],
                         ['ING', 'ABN'],
                         ['ABN', 'BNP'],
                         ['BNP', 'ING']],
                         columns=['creditor', 'debtor'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id='name', src='creditor', dst='debtor')

            msg = 'There are repeated edges'
            assert e_info.value.args[0] == msg

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame([['ING', 'ABN'],
                         ['BNP', 'ABN'],
                         ['RAB', 'ABN'],
                         ['ABN', 'BNP'],
                         ['BNP', 'ING']],
                         columns=['creditor', 'debtor'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id='name', src='creditor', dst='debtor')

            msg = 'Some source vertices are not in v.'
            assert e_info.value.args[0] == msg

        e = pd.DataFrame([['ING', 'ABN'],
                         ['BNP', 'ABN'],
                         ['ING', 'RAB'],
                         ['ABN', 'BNP'],
                         ['BNP', 'ING']],
                         columns=['creditor', 'debtor'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id='name', src='creditor', dst='debtor')

            msg = 'Some destination vertices are not in v.'
            assert e_info.value.args[0] == msg

    def test_degree_init(self):
        v = pd.DataFrame([['ING'], ['ABN'], ['BNP'], ['RAB'], ['UBS']],
                         columns=['name'])
        d = np.array([2, 2, 2, 0, 0])

        with pytest.warns(UserWarning):
            g = ge.Graph(v, self.e, v_id='name', src='creditor', dst='debtor')

            assert np.all(g.v.degree == d), g.v.degree

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame([['ING'], ['ABN'], ['BNP'], ['RAB']],
                         columns=['name'])

        with pytest.warns(UserWarning, match='RAB vertex has no edges.'):
            ge.Graph(v, self.e, v_id='name', src='creditor', dst='debtor')

        v = pd.DataFrame([['ING'], ['ABN'], ['BNP'], ['RAB'], ['UBS']],
                         columns=['name'])

        with pytest.warns(UserWarning, match=r' vertices have no edges.'):
            ge.Graph(v, self.e, v_id='name', src='creditor', dst='debtor')

    def test_out_degree(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')
        d_out = np.array([1, 1, 2])
        d_test = g.out_degree(get=True)

        assert np.all(d_test == d_out), d_test
        assert np.all(g.v.out_degree == d_out), g.v.out_degree

    def test_in_degree(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')
        d_in = np.array([1, 2, 1])
        d_test = g.in_degree(get=True)

        assert np.all(d_test == d_in), d_test
        assert np.all(g.v.in_degree == d_in), g.v.in_degree


class TestWeightedGraph():
    v = pd.DataFrame([['ING', 'NL', 1e12],
                     ['ABN', 'NL', 5e11],
                     ['BNP', 'FR', 13e12]],
                     columns=['name', 'country', 'assets'])

    e = pd.DataFrame([['ING', 'ABN', 1e6],
                     ['BNP', 'ABN', 1.7e5],
                     ['ABN', 'BNP', 1e4]],
                     columns=['creditor', 'debtor', 'value'])

    _e = np.sort(np.rec.array([(0, 1, 1e6), (2, 1, 1.7e5), (1, 2, 1e4)],
                              dtype=[('src', np.uint8),
                                     ('dst', np.uint8),
                                     ('weight', np.float64)]))

    def test_init(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.WeightedGraph)
        assert np.all(g.e == self._e), g.e

# class TestSimpleGraph():
#     v = pd.DataFrame([['ING', 'NL', 1e12],
#                      ['ABN', 'NL', 5e11],
#                      ['BNP', 'FR', 13e12]],
#                      columns=['name', 'country', 'assets'])

#     e = pd.DataFrame([['ING', 'ABN', 1e6, 'interbank'],
#                      ['BNP', 'ABN', 2.3e7, 'external'],
#                      ['BNP', 'ABN', 1.7e5, 'interbank'],
#                      ['ABN', 'BNP', 1e4, 'interbank'],
#                      ['ABN', 'ING', 4e5, 'external']],
#                      columns=['creditor', 'debtor', 'value', 'type'])

#     def test_wrong_input(self):
#         with pytest.raises(Exception) as e_info:
#             ge.Graph([[1], [2], [3]], [[1, 2], [3, 1]])

#         msg = 'Only dataframe input supported.'
#         assert e_info.value.args[0] == msg

#     def test_instanciation_names(self):
#         g = ge.Graph(self.v, self.e, id_col='name', src_col='creditor',
#                      dst_col='debtor')

#         assert isinstance(g, ge.Graph)
