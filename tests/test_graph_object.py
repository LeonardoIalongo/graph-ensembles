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

    def test_init(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')
        test_e = np.sort(np.rec.array([(0, 1), (2, 1), (1, 2), (2, 0)],
                         dtype=[('src', np.uint8), ('dst', np.uint8)]))

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.DirectedGraph)
        assert np.all(g.e == test_e), g.e

    def test_init_id(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')
        test_dict = {'ING': 0, 'ABN': 1, 'BNP': 2}
        assert test_dict == g.id_dict

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

        msg = 'There are repeated edges.'
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

    def test_vertex_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name',
                     src='creditor', dst='debtor', v_group='country')

        test_v = np.array([0, 0, 1], dtype=np.uint8)

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.DirectedGraph)
        assert np.all(g.v.group == test_v), g.v.group

    def test_degree_init(self):
        v = pd.DataFrame([['ING'], ['ABN'], ['BNP'], ['RAB'], ['UBS']],
                         columns=['name'])
        d = np.array([2, 2, 2, 0, 0])

        with pytest.warns(UserWarning):
            g = ge.Graph(v, self.e, v_id='name', src='creditor', dst='debtor')

        assert np.all(g.v.degree == d), g.v.degree

    def test_degree_by_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name',
                     src='creditor', dst='debtor', v_group='country')

        d = np.rec.array([(0, 0, 1),
                         (0, 1, 1),
                         (1, 0, 1),
                         (1, 1, 2),
                         (2, 0, 3)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.uint8)])

        d_test = g.degree_by_group(get=True)

        assert np.all(d_test == d), d_test
        assert np.all(g.gv.degree == d), g.gv.degree

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

    def test_out_degree_by_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name',
                     src='creditor', dst='debtor', v_group='country')

        d = np.rec.array([(0, 0, 1),
                         (1, 1, 1),
                         (2, 0, 2)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.uint8)])

        d_test = g.out_degree_by_group(get=True)

        assert np.all(d_test == d), d_test
        assert np.all(g.gv.out_degree == d), g.gv.out_degree

    def test_in_degree_by_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name',
                     src='creditor', dst='debtor', v_group='country')

        d = np.rec.array([(0, 1, 1),
                         (1, 0, 1),
                         (1, 1, 1),
                         (2, 0, 1)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.uint8)])

        d_test = g.in_degree_by_group(get=True)

        assert np.all(d_test == d), d_test
        assert np.all(g.gv.in_degree == d), g.gv.in_degree

    def test_multi_id_init(self):
        v = pd.DataFrame([['ING', 'NL'],
                         ['ABN', 'NL'],
                         ['BNP', 'FR'],
                         ['BNP', 'IT']],
                         columns=['name', 'country'])

        e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL'],
                         ['BNP', 'FR', 'ABN', 'NL'],
                         ['ABN', 'NL', 'BNP', 'IT'],
                         ['BNP', 'IT', 'ING', 'NL']],
                         columns=['creditor', 'c_country',
                                  'debtor', 'd_country'])

        g = ge.Graph(v, e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'])

        test_e = np.sort(np.rec.array([(0, 1), (2, 1), (1, 3), (3, 0)],
                         dtype=[('src', np.uint8), ('dst', np.uint8)]))
        test_dict = {('ING', 'NL'): 0, ('ABN', 'NL'): 1,
                     ('BNP', 'FR'): 2, ('BNP', 'IT'): 3}

        assert np.all(g.e == test_e), g.e
        assert test_dict == g.id_dict

    def test_to_sparse(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')

        mat = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0]])
        assert np.all(g.adjacency_matrix().toarray() == mat)

    def test_to_networkx(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')

        v = [0, 1, 2]
        e = [(0, 1), (1, 2), (2, 0), (2, 1)]
        adj = {0: {1: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}

        gx = g.to_networkx()
        assert np.all(list(gx.nodes) == v)
        assert np.all(list(gx.edges) == e)
        assert gx.adj == adj

    def test_to_networkx_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name',
                     src='creditor', dst='debtor', v_group='country')

        v = [(0, {'group': 0}), (1, {'group': 0}), (2, {'group': 1})]
        e = [(0, 1, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]
        adj = {0: {1: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj

    def test_to_networkx_orig(self):
        g = ge.Graph(self.v, self.e, v_id='name',
                     src='creditor', dst='debtor')

        v = ['ING', 'ABN', 'BNP']
        e = [('ING', 'ABN'), ('ABN', 'BNP'), ('BNP', 'ING'), ('BNP', 'ABN')]
        adj = {'ING': {'ABN': {}},
               'ABN': {'BNP': {}},
               'BNP': {'ING': {}, 'ABN': {}}}

        gx = g.to_networkx(original=True)
        assert np.all(list(gx.nodes) == v)
        assert np.all(list(gx.edges) == e)
        assert gx.adj == adj


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

    def test_vertex_group(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value', v_group='country')

        test_v = np.array([0, 0, 1], dtype=np.uint8)

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.WeightedGraph)
        assert np.all(g.v.group == test_v), g.v.group

    def test_total_weight(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')

        assert g.total_weight == 1e6 + 1.7e5 + 1e4

    def test_strength(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')
        s = np.array([1e6, 1e6 + 1.7e5 + 1e4, 1.7e5 + 1e4])
        s_test = g.strength(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.v.strength == s), g.v.strength

    def test_out_strength(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')
        s_out = np.array([1e6, 1e4, 1.7e5])
        s_test = g.out_strength(get=True)

        assert np.all(s_test == s_out), s_test
        assert np.all(g.v.out_strength == s_out), g.v.out_strength

    def test_in_strength(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')
        s_in = np.array([0, 1e6 + 1.7e5, 1e4])
        s_test = g.in_strength(get=True)

        assert np.all(s_test == s_in), s_test
        assert np.all(g.v.in_strength == s_in), g.v.in_strength

    def test_strength_by_group(self):
        g = ge.Graph(self.v, self.e, v_id='name', weight='value',
                     src='creditor', dst='debtor', v_group='country')

        s = np.rec.array([(0, 0, 1e6),
                         (1, 0, 1e6),
                         (1, 1, 1.7e5 + 1e4),
                         (2, 0, 1.7e5 + 1e4)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.float64)])

        s_test = g.strength_by_group(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.gv.strength == s), g.gv.strength

    def test_out_strength_by_group(self):
        g = ge.Graph(self.v, self.e, v_id='name', weight='value',
                     src='creditor', dst='debtor', v_group='country')

        s = np.rec.array([(0, 0, 1e6),
                         (1, 1, 1e4),
                         (2, 0, 1.7e5)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.float64)])

        s_test = g.out_strength_by_group(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.gv.out_strength == s), g.gv.out_strength

    def test_in_strength_by_group(self):
        g = ge.Graph(self.v, self.e, v_id='name', weight='value',
                     src='creditor', dst='debtor', v_group='country')

        s = np.rec.array([(1, 0, 1e6),
                         (1, 1, 1.7e5),
                         (2, 0, 1e4)],
                         dtype=[('id', np.uint8),
                                ('group', np.uint8),
                                ('value', np.float64)])

        s_test = g.in_strength_by_group(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.gv.in_strength == s), g.gv.in_strength

    def test_to_sparse(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')

        mat = np.array([[0, 1e6, 0],
                        [0, 0, 1e4],
                        [0, 1.7e5, 0]])
        assert np.all(g.adjacency_matrix().toarray() == mat)

    def test_to_networkx(self):
        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', weight='value')

        v = [0, 1, 2]
        e = [(0, 1), (1, 2), (2, 1)]
        adj = {0: {1: {'weight': 1e6}},
               1: {2: {'weight': 1e4}},
               2: {1: {'weight': 1.7e5}}}

        gx = g.to_networkx()
        assert np.all(list(gx.nodes) == v)
        assert np.all(list(gx.edges) == e)
        assert gx.adj == adj

    def test_to_networkx_group(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(v, self.e, v_id='name', v_group='country',
                     src='creditor', dst='debtor', weight='value')

        v = [(0, {'group': 0}), (1, {'group': 0}), (2, {'group': 1})]
        e = [(0, 1, {'weight': 1e6}),
             (1, 2, {'weight': 1e4}),
             (2, 1, {'weight': 1.7e5})]
        adj = {0: {1: {'weight': 1e6}},
               1: {2: {'weight': 1e4}},
               2: {1: {'weight': 1.7e5}}}

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj

    def test_to_networkx_orig(self):
        v = pd.DataFrame([['ING', 'NL'], ['ABN', 'NL'], ['BNP', 'FR']],
                         columns=['name', 'country'])

        g = ge.Graph(self.v, self.e, v_id='name', src='creditor',
                     dst='debtor', v_group='country', weight='value')

        v = [('ING', {'group': 'NL'}),
             ('ABN', {'group': 'NL'}),
             ('BNP', {'group': 'FR'})]
        e = [('ING', 'ABN', {'weight': 1e6}),
             ('ABN', 'BNP', {'weight': 1e4}),
             ('BNP', 'ABN', {'weight': 1.7e5})]
        adj = {'ING': {'ABN': {'weight': 1e6}},
               'ABN': {'BNP': {'weight': 1e4}},
               'BNP': {'ABN': {'weight': 1.7e5}}}

        gx = g.to_networkx(original=True)
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj


class TestLabelGraph():
    v = pd.DataFrame([['ING', 'NL'],
                     ['ABN', 'NL'],
                     ['BNP', 'FR'],
                     ['BNP', 'IT']],
                     columns=['name', 'country'])

    e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                     ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                     ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                     ['BNP', 'IT', 'ABN', 'NL', 3e3, 'interbank', False],
                     ['ABN', 'NL', 'BNP', 'FR', 1e4, 'interbank', False],
                     ['ABN', 'NL', 'ING', 'NL', 4e5, 'external', True]],
                     columns=['creditor', 'c_country',
                              'debtor', 'd_country',
                              'value', 'type', 'EUR'])
    v_s = pd.DataFrame([['ING', 'NL', 1e12],
                       ['ABN', 'NL', 5e11],
                       ['BNP', 'FR', 13e12]],
                       columns=['name', 'country', 'assets'])

    e_s = pd.DataFrame([['ING', 'ABN', 1e6, 'interbank'],
                       ['BNP', 'ABN', 2.3e7, 'external'],
                       ['BNP', 'ABN', 1.7e5, 'interbank'],
                       ['ABN', 'BNP', 1e4, 'interbank'],
                       ['ABN', 'ING', 4e5, 'external']],
                       columns=['creditor', 'debtor', 'value', 'type'])

    def test_init_class(self):
        g = ge.Graph(self.v_s, self.e_s, v_id='name', src='creditor',
                     dst='debtor', edge_label='type')

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.LabelGraph)
        assert g.num_edges == 4

    def test_init_edges(self):
        g = ge.Graph(self.v_s, self.e_s, v_id='name', src='creditor',
                     dst='debtor', edge_label='type')
        test_e = np.sort(np.rec.array([(0, 0, 1),
                                       (1, 2, 1),
                                       (0, 2, 1),
                                       (0, 1, 2),
                                       (1, 1, 0)],
                         dtype=[('label', np.uint8),
                                ('src', np.uint8),
                                ('dst', np.uint8)]))

        assert np.all(g.e == test_e), g.e

    def test_init_id(self):
        g = ge.Graph(self.v_s, self.e_s, v_id='name', src='creditor',
                     dst='debtor', edge_label='type')
        test_dict = {'ING': 0, 'ABN': 1, 'BNP': 2}
        assert test_dict == g.id_dict

    def test_init_label(self):
        g = ge.Graph(self.v_s, self.e_s, v_id='name', src='creditor',
                     dst='debtor', edge_label='type')
        test_dict = {'interbank': 0, 'external': 1}
        assert test_dict == g.label_dict

    def test_multi_id_init(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        test_e = np.rec.array([(0, 0, 1),
                               (1, 2, 1),
                               (2, 3, 1),
                               (0, 3, 1),
                               (0, 1, 2),
                               (3, 1, 0)],
                              dtype=[('label', np.uint8),
                                     ('src', np.uint8),
                                     ('dst', np.uint8)])
        test_e.sort()
        test_dict = {('ING', 'NL'): 0, ('ABN', 'NL'): 1,
                     ('BNP', 'FR'): 2, ('BNP', 'IT'): 3}
        test_label = {('interbank', False): 0, ('external', False): 1,
                      ('interbank', True): 2, ('external', True): 3}

        assert np.all(g.e == test_e), g.e
        assert test_dict == g.id_dict, g.id_dict
        assert test_label == g.label_dict, g.label_dict

    def test_duplicated_vertices(self):
        v = pd.DataFrame([['ING', 'NL'],
                         ['ABN', 'NL'],
                         ['BNP', 'FR'],
                         ['ABN', 'NL'],
                         ['BNP', 'IT']],
                         columns=['name', 'country'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        msg = 'There is at least one repeated id in the vertex dataframe.'
        assert e_info.value.args[0] == msg

    def test_duplicated_edges(self):
        e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                         ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                         ['ABN', 'NL', 'BNP', 'FR', 1e4, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 4e5, 'external', False]],
                         columns=['creditor', 'c_country',
                                  'debtor', 'd_country',
                                  'value', 'type', 'EUR'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        msg = 'There are repeated edges.'
        assert e_info.value.args[0] == msg

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                         ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                         ['ABN', 'UK', 'BNP', 'FR', 1e4, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 4e5, 'external', False]],
                         columns=['creditor', 'c_country',
                                  'debtor', 'd_country',
                                  'value', 'type', 'EUR'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        msg = 'Some source vertices are not in v.'
        assert e_info.value.args[0] == msg

        e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                         ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                         ['ABN', 'NL', 'UBS', 'FR', 1e4, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 4e5, 'external', False]],
                         columns=['creditor', 'c_country',
                                  'debtor', 'd_country',
                                  'value', 'type', 'EUR'])

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        msg = 'Some destination vertices are not in v.'
        assert e_info.value.args[0] == msg

    def test_vertex_group(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     v_group='country')

        test_v = np.array([0, 0, 1, 2], dtype=np.uint8)

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.LabelGraph)
        assert np.all(g.v.group == test_v), g.v.group

    def test_degree_init(self):
        v = pd.DataFrame([['ING', 'NL'],
                         ['ABN', 'NL'],
                         ['BNP', 'FR'],
                         ['BNP', 'IT'],
                         ['ABN', 'UK'],
                         ['UBS', 'FR']],
                         columns=['name', 'country'])

        d = np.array([1, 3, 1, 1, 0, 0])

        with pytest.warns(UserWarning):
            g = ge.Graph(v, self.e, v_id=['name', 'country'],
                         src=['creditor', 'c_country'],
                         dst=['debtor', 'd_country'],
                         edge_label=['type', 'EUR'])

        assert np.all(g.v.degree == d), g.v.degree

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame([['ING', 'NL'],
                         ['ABN', 'NL'],
                         ['BNP', 'FR'],
                         ['BNP', 'IT'],
                         ['ABN', 'UK'],
                         ['UBS', 'FR']],
                         columns=['name', 'country'])

        with pytest.warns(UserWarning, match=r' vertices have no edges.'):
            ge.Graph(v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

    def test_out_degree(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])
        d_out = np.array([1, 2, 1, 1])
        d_test = g.out_degree(get=True)

        assert np.all(d_test == d_out), d_test
        assert np.all(g.v.out_degree == d_out), g.v.out_degree

    def test_in_degree(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])
        d_in = np.array([1, 3, 1, 0])
        d_test = g.in_degree(get=True)

        assert np.all(d_test == d_in), d_test
        assert np.all(g.v.in_degree == d_in), g.v.in_degree

    def test_num_edges_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])
        test_num = np.array([3, 1, 1, 1], dtype='u1')
        assert np.all(g.num_edges_label == test_num)

    def test_degree_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        d = np.rec.array([(0, 0, 1),
                         (0, 1, 3),
                         (0, 2, 1),
                         (0, 3, 1),
                         (1, 1, 1),
                         (1, 2, 1),
                         (2, 1, 1),
                         (2, 3, 1),
                         (3, 0, 1),
                         (3, 1, 1)],
                         dtype=[('label', np.uint8),
                                ('id', np.uint8),
                                ('value', np.uint8)])

        d_test = g.degree_by_label(get=True)

        assert np.all(d_test == d), d_test
        assert np.all(g.lv.degree == d), g.lv.degree

    def test_out_degree_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        d_out = np.rec.array([(0, 0, 1),
                             (0, 1, 1),
                             (0, 3, 1),
                             (1, 2, 1),
                             (2, 3, 1),
                             (3, 1, 1)],
                             dtype=[('label', np.uint8),
                                    ('id', np.uint8),
                                    ('value', np.uint8)])

        d_test = g.out_degree_by_label(get=True)

        assert np.all(d_test == d_out), d_test
        assert np.all(g.lv.out_degree == d_out), g.lv.out_degree

    def test_in_degree_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        d_in = np.rec.array([(0, 1, 2),
                             (0, 2, 1),
                             (1, 1, 1),
                             (2, 1, 1),
                             (3, 0, 1)],
                            dtype=[('label', np.uint8),
                                   ('id', np.uint8),
                                   ('value', np.uint8)])

        d_test = g.in_degree_by_label(get=True)

        assert np.all(d_test == d_in), d_test
        assert np.all(g.lv.in_degree == d_in), g.lv.in_degree

    def test_to_sparse_compressed(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        mat = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]], dtype=float)
        assert np.all(g.adjacency_matrix(compressed=True).toarray() == mat)

    def test_to_sparse(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        mat = [np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])]

        for i in range(len(mat)):
            assert np.all(g.adjacency_matrix()[i].toarray() == mat[i])

    def test_to_networkx(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'])

        v = [0, 1, 2, 3]
        e = [(0, 1, {'label': 0}),
             (1, 2, {'label': 0}),
             (1, 0, {'label': 3}),
             (2, 1, {'label': 1}),
             (3, 1, {'label': 0}),
             (3, 1, {'label': 2})]

        gx = g.to_networkx()
        assert np.all(list(gx.nodes) == v)
        assert np.all(list(gx.edges(data=True)) == e)

    def test_to_networkx_group(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     v_group='country')

        v = [(0, {'group': 0}),
             (1, {'group': 0}),
             (2, {'group': 1}),
             (3, {'group': 2})]
        e = [(0, 1, {'label': 0}),
             (1, 2, {'label': 0}),
             (1, 0, {'label': 3}),
             (2, 1, {'label': 1}),
             (3, 1, {'label': 0}),
             (3, 1, {'label': 2})]

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)

    def test_to_networkx_orig(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     v_group='country')

        v = [(('ING', 'NL'), {'group': 'NL'}),
             (('ABN', 'NL'), {'group': 'NL'}),
             (('BNP', 'FR'), {'group': 'FR'}),
             (('BNP', 'IT'), {'group': 'IT'})]
        e = [(('ING', 'NL'), ('ABN', 'NL'), {'label': ('interbank', False)}),
             (('ABN', 'NL'), ('BNP', 'FR'), {'label': ('interbank', False)}),
             (('ABN', 'NL'), ('ING', 'NL'), {'label': ('external', True)}),
             (('BNP', 'FR'), ('ABN', 'NL'), {'label': ('external', False)}),
             (('BNP', 'IT'), ('ABN', 'NL'), {'label': ('interbank', False)}),
             (('BNP', 'IT'), ('ABN', 'NL'), {'label': ('interbank', True)})]

        gx = g.to_networkx(original=True)
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)


class TestWeightedLabelGraph():
    v = pd.DataFrame([['ING', 'NL', 'HQ'],
                     ['ABN', 'NL', 'HQ'],
                     ['BNP', 'FR', 'HQ'],
                     ['BNP', 'IT', 'branch']],
                     columns=['name', 'country', 'legal'])

    e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                     ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                     ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                     ['BNP', 'IT', 'ABN', 'NL', 3e3, 'interbank', False],
                     ['ABN', 'NL', 'BNP', 'FR', 1e4, 'interbank', False],
                     ['ABN', 'NL', 'ING', 'NL', 4e5, 'external', True]],
                     columns=['creditor', 'c_country',
                              'debtor', 'd_country',
                              'value', 'type', 'EUR'])

    def test_init(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')
        test_e = np.rec.array([(0, 0, 1, 1e6),
                               (1, 2, 1, 2.3e7),
                               (2, 3, 1, 7e5),
                               (0, 3, 1, 3e3),
                               (0, 1, 2, 1e4),
                               (3, 1, 0, 4e5)],
                              dtype=[('label', np.uint8),
                                     ('src', np.uint8),
                                     ('dst', np.uint8),
                                     ('weight', np.float64)])
        test_e.sort()

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.WeightedLabelGraph)
        assert np.all(g.e == test_e), g.e

    def test_vertex_group(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value',
                     v_group=['country', 'legal'])

        test_v = np.array([0, 0, 1, 2], dtype=np.uint8)

        assert isinstance(g, ge.sGraph)
        assert isinstance(g, ge.WeightedLabelGraph)
        assert np.all(g.v.group == test_v), g.v.group

    def test_total_weight(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        assert g.total_weight == 1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5

    def test_strength(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')
        s = np.array([1e6 + 4e5,
                      1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5,
                      2.3e7 + 1e4,
                      7e5 + 3e3])
        s_test = g.strength(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.v.strength == s), g.v.strength

    def test_out_strength(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')
        s_out = np.array([1e6, 1e4 + 4e5, 2.3e7, 7e5 + 3e3])
        s_test = g.out_strength(get=True)

        assert np.all(s_test == s_out), s_test
        assert np.all(g.v.out_strength == s_out), g.v.out_strength

    def test_in_strength(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')
        s_in = np.array([4e5, 1e6 + 2.3e7 + 7e5 + 3e3, 1e4, 0])
        s_test = g.in_strength(get=True)

        assert np.all(s_test == s_in), s_test
        assert np.all(g.v.in_strength == s_in), g.v.in_strength

    def test_total_weight_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        w_by_l = np.array([1e6 + 3e3 + 1e4, 2.3e7, 7e5, 4e5], dtype='f8')
        test = g.total_weight_label
        assert np.all(test == w_by_l), test

    def test_strength_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        s = np.rec.array([(0, 0, 1e6),
                         (0, 1, 1e4 + 1e6 + 3e3),
                         (0, 2, 1e4),
                         (0, 3, 3e3),
                         (1, 1, 2.3e7),
                         (1, 2, 2.3e7),
                         (2, 1, 7e5),
                         (2, 3, 7e5),
                         (3, 0, 4e5),
                         (3, 1, 4e5)],
                         dtype=[('label', np.uint8),
                                ('id', np.uint8),
                                ('value', np.float64)])

        s_test = g.strength_by_label(get=True)

        assert np.all(s_test == s), s_test
        assert np.all(g.lv.strength == s), g.v.strength

    def test_out_strength_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        s_out = np.rec.array([(0, 0, 1e6),
                             (0, 1, 1e4),
                             (0, 3, 3e3),
                             (1, 2, 2.3e7),
                             (2, 3, 7e5),
                             (3, 1, 4e5)],
                             dtype=[('label', np.uint8),
                                    ('id', np.uint8),
                                    ('value', np.float64)])

        s_test = g.out_strength_by_label(get=True)

        assert np.all(s_test == s_out), s_test
        assert np.all(g.lv.out_strength == s_out), g.lv.out_strength

    def test_in_strength_by_label(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        s_in = np.rec.array([(0, 1, 1e6 + 3e3),
                             (0, 2, 1e4),
                             (1, 1, 2.3e7),
                             (2, 1, 7e5),
                             (3, 0, 4e5)],
                            dtype=[('label', np.uint8),
                                   ('id', np.uint8),
                                   ('value', np.float64)])

        s_test = g.in_strength_by_label(get=True)

        assert np.all(s_test == s_in), s_test
        assert np.all(g.lv.in_strength == s_in), g.v.in_strength

    def test_to_sparse_compressed(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        mat = np.array([[0, 1e6, 0, 0],
                        [4e5, 0, 1e4, 0],
                        [0, 2.3e7, 0, 0],
                        [0, 7e5 + 3e3, 0, 0]])
        assert np.all(g.adjacency_matrix(compressed=True).toarray() == mat)

    def test_to_sparse(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        mat = [np.array([[0, 1e6, 0, 0],
                        [0, 0, 1e4, 0],
                        [0, 0, 0, 0],
                        [0, 3e3, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 2.3e7, 0, 0],
                        [0, 0, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 7e5, 0, 0]]),
               np.array([[0, 0, 0, 0],
                        [4e5, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])]

        for i in range(len(mat)):
            assert np.all(g.adjacency_matrix()[i].toarray() == mat[i])

    def test_to_networkx(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        v = [0, 1, 2, 3]
        e = [(0, 1, {'weight': 1e6, 'label': 0}),
             (1, 2, {'weight': 1e4, 'label': 0}),
             (1, 0, {'weight': 4e5, 'label': 3}),
             (2, 1, {'weight': 2.3e7, 'label': 1}),
             (3, 1, {'weight': 3e3, 'label': 0}),
             (3, 1, {'weight': 7e5, 'label': 2})]

        gx = g.to_networkx()
        assert np.all(list(gx.nodes) == v)
        assert np.all(list(gx.edges(data=True)) == e)

    def test_to_networkx_group(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value', v_group='legal')

        v = [(0, {'group': 0}),
             (1, {'group': 0}),
             (2, {'group': 0}),
             (3, {'group': 1})]
        e = [(0, 1, {'weight': 1e6, 'label': 0}),
             (1, 2, {'weight': 1e4, 'label': 0}),
             (1, 0, {'weight': 4e5, 'label': 3}),
             (2, 1, {'weight': 2.3e7, 'label': 1}),
             (3, 1, {'weight': 3e3, 'label': 0}),
             (3, 1, {'weight': 7e5, 'label': 2})]

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)

    def test_to_networkx_orig(self):
        g = ge.Graph(self.v, self.e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value', v_group='country')

        v = [(('ING', 'NL'), {'group': 'NL'}),
             (('ABN', 'NL'), {'group': 'NL'}),
             (('BNP', 'FR'), {'group': 'FR'}),
             (('BNP', 'IT'), {'group': 'IT'})]
        e = [(('ING', 'NL'), ('ABN', 'NL'),
              {'weight': 1e6, 'label': ('interbank', False)}),
             (('ABN', 'NL'), ('BNP', 'FR'),
              {'weight': 1e4, 'label': ('interbank', False)}),
             (('ABN', 'NL'), ('ING', 'NL'),
              {'weight': 4e5, 'label': ('external', True)}),
             (('BNP', 'FR'), ('ABN', 'NL'),
              {'weight': 2.3e7, 'label': ('external', False)}),
             (('BNP', 'IT'), ('ABN', 'NL'),
              {'weight': 3e3, 'label': ('interbank', False)}),
             (('BNP', 'IT'), ('ABN', 'NL'),
              {'weight': 7e5, 'label': ('interbank', True)})]

        gx = g.to_networkx(original=True)
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
