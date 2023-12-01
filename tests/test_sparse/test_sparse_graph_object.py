""" Test graph class object creation and attributes. """
import graph_ensembles.sparse as ge
import pandas as pd
import numpy as np
import pytest


class TestGraph:
    v = pd.DataFrame(
        [["ING", "NL", 1e12], ["ABN", "NL", 5e11], ["BNP", "FR", 13e12]],
        columns=["name", "country", "assets"],
    )

    e = pd.DataFrame(
        [
            ["ING", "ABN", 1e6],
            ["BNP", "ABN", 1.7e5],
            ["ABN", "BNP", 1e4],
            ["BNP", "ING", 3e3],
        ],
        columns=["creditor", "debtor", "value"],
    )

    adj = np.zeros((3, 3), dtype=np.float64)
    adj[2, 0] = 1e6
    adj[1, 0] = 1.7e5
    adj[0, 1] = 1e4
    adj[1, 2] = 3e3

    def test_init(self):
        g = ge.Graph(self.v, self.e, v_id="name", src="creditor", dst="debtor")

        assert isinstance(g, ge.Graph)

        adj = self.adj != 0
        adj = adj | adj.T
        res = g.adjacency_matrix().todense()

        assert np.all(res == adj), res

    def test_init_weights(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )

        assert isinstance(g, ge.Graph)

        adj = self.adj + self.adj.T
        res = g.adjacency_matrix(weighted=True).todense()

        assert np.all(res == adj), res

    def test_init_id(self):
        g = ge.Graph(self.v, self.e, v_id="name", src="creditor", dst="debtor")
        test_dict = {"ABN": 0, "BNP": 1, "ING": 2}
        assert test_dict == g.id_dict

    def test_duplicated_vertices(self):
        v = pd.DataFrame([["ING"], ["ABN"], ["BNP"], ["ABN"]], columns=["name"])

        with pytest.raises(Exception) as e_info:
            ge.Graph(v, self.e, v_id="name", src="creditor", dst="debtor")

        msg = "There is at least one repeated id in the vertex dataframe."
        assert e_info.value.args[0] == msg

    def test_duplicated_edges(self):
        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["ING", "ABN"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        g = ge.Graph(self.v, e, v_id="name", src="creditor", dst="debtor")

        assert isinstance(g, ge.Graph)

        adj = self.adj != 0
        adj = adj | adj.T
        res = g.adjacency_matrix().todense()

        assert np.all(res == adj), res

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["RAB", "ABN"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id="name", src="creditor", dst="debtor")

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["ING", "RAB"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        with pytest.raises(Exception) as e_info:
            ge.Graph(self.v, e, v_id="name", src="creditor", dst="debtor")

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

    def test_vertex_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.Graph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        test_v = np.array([1, 0, 1], dtype=np.uint8)

        assert isinstance(g, ge.Graph)
        assert np.all(g.groups == test_v), g.groups

    def test_num_edges(self):
        g = ge.Graph(self.v, self.e, v_id="name", src="creditor", dst="debtor")

        assert g.num_edges() == 3

    def test_degree_init(self):
        v = pd.DataFrame(
            [["ING"], ["ABN"], ["BNP"], ["RAB"], ["UBS"]], columns=["name"]
        )
        d = np.array([2, 2, 2, 0, 0])

        with pytest.warns(UserWarning):
            g = ge.Graph(v, self.e, v_id="name", src="creditor", dst="debtor")

        assert np.all(g.degree() == d), g.degree()

    def test_total_weight(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )

        assert g.total_weight() == 1e6 + 1.7e5 + 1e4 + 3e3

    def test_strength(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        s = np.array([1e6 + 1.7e5 + 1e4, 1.7e5 + 1e4 + 3e3, 1e6 + 3e3])
        s_test = g.strength()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength == s), g._strength

    def test_degree_by_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.Graph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        d = np.array([[1, 1], [0, 2], [1, 1]])

        d_test = g.degree_by_group().todense()

        assert np.all(d_test == d), d_test
        assert np.all(g._degree_by_group.todense() == d), d_test

    def test_strength_by_group(self):
        g = ge.Graph(
            self.v,
            self.e,
            v_id="name",
            weight="value",
            src="creditor",
            dst="debtor",
            v_group="country",
        )

        s = np.array([(1.7e5 + 1e4, 1e6), (0, 1.7e5 + 1e4 + 3e3), (3e3, 1e6)])
        s_test = g.strength_by_group()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength_by_group == s), g._strength_by_group

    def test_av_nn_prop_ones(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.ones(g.num_vertices)
        avnn = g.average_nn_property(prop)

        assert np.all(avnn == prop), avnn

    def test_av_nn_prop_zeros(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.zeros(g.num_vertices)
        avnn = g.average_nn_property(prop)

        assert np.all(avnn == 0), avnn

    def test_av_nn_prop_scale(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.arange(g.num_vertices)
        avnn = g.average_nn_property(prop)
        test = np.array([1.5, 1.0, 0.5])

        assert np.all(avnn == test), avnn

    def test_av_nn_deg(self):
        g = ge.Graph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        avnn = g.average_nn_degree()
        test = np.array([2, 2, 2])

        assert np.all(avnn == test), avnn

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame([["ING"], ["ABN"], ["BNP"], ["RAB"]], columns=["name"])

        with pytest.warns(UserWarning, match="RAB vertex has no edges."):
            ge.Graph(v, self.e, v_id="name", src="creditor", dst="debtor")

        v = pd.DataFrame(
            [["ING"], ["ABN"], ["BNP"], ["RAB"], ["UBS"]], columns=["name"]
        )

        with pytest.warns(UserWarning, match=r" vertices have no edges."):
            ge.Graph(v, self.e, v_id="name", src="creditor", dst="debtor")

    def test_to_networkx(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.Graph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        v = [
            (0, {"node_id": "ABN", "group": "NL"}),
            (1, {"node_id": "BNP", "group": "FR"}),
            (2, {"node_id": "ING", "group": "NL"}),
        ]
        e = [
            (0, 1, {"weight": True}),
            (0, 2, {"weight": True}),
            (1, 2, {"weight": True}),
        ]
        adj = {
            0: {1: {"weight": True}, 2: {"weight": True}},
            1: {0: {"weight": True}, 2: {"weight": True}},
            2: {0: {"weight": True}, 1: {"weight": True}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj

    def test_to_networkx_weights(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.Graph(
            v,
            self.e,
            v_id="name",
            src="creditor",
            dst="debtor",
            v_group="country",
            weight="value",
        )

        v = [
            (0, {"node_id": "ABN", "group": "NL"}),
            (1, {"node_id": "BNP", "group": "FR"}),
            (2, {"node_id": "ING", "group": "NL"}),
        ]
        e = [
            (0, 1, {"weight": 1e4 + 1.7e5}),
            (0, 2, {"weight": 1e6}),
            (1, 2, {"weight": 3e3}),
        ]
        adj = {
            0: {1: {"weight": 1e4 + 1.7e5}, 2: {"weight": 1e6}},
            1: {0: {"weight": 1e4 + 1.7e5}, 2: {"weight": 3e3}},
            2: {0: {"weight": 1e6}, 1: {"weight": 3e3}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj


class TestDiGraph:
    v = pd.DataFrame(
        [["ING", "NL", 1e12], ["ABN", "NL", 5e11], ["BNP", "FR", 13e12]],
        columns=["name", "country", "assets"],
    )

    e = pd.DataFrame(
        [
            ["ING", "ABN", 1e6],
            ["BNP", "ABN", 1.7e5],
            ["ABN", "BNP", 1e4],
            ["BNP", "ING", 3e3],
        ],
        columns=["creditor", "debtor", "value"],
    )

    adj = np.zeros((3, 3), dtype=np.float64)
    adj[2, 0] = 1e6
    adj[1, 0] = 1.7e5
    adj[0, 1] = 1e4
    adj[1, 2] = 3e3

    def test_init(self):
        g = ge.DiGraph(self.v, self.e, v_id="name", src="creditor", dst="debtor")

        assert isinstance(g, ge.DiGraph)

        adj = self.adj != 0
        res = g.adjacency_matrix().todense()

        assert np.all(res == adj), res

    def test_init_weights(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )

        assert isinstance(g, ge.Graph)

        res = g.adjacency_matrix(weighted=True).todense()

        assert np.all(res == self.adj), res

    def test_init_id(self):
        g = ge.DiGraph(self.v, self.e, v_id="name", src="creditor", dst="debtor")
        test_dict = {"ABN": 0, "BNP": 1, "ING": 2}
        assert test_dict == g.id_dict

    def test_duplicated_vertices(self):
        v = pd.DataFrame([["ING"], ["ABN"], ["BNP"], ["ABN"]], columns=["name"])

        with pytest.raises(Exception) as e_info:
            ge.DiGraph(v, self.e, v_id="name", src="creditor", dst="debtor")

        msg = "There is at least one repeated id in the vertex dataframe."
        assert e_info.value.args[0] == msg

    def test_duplicated_edges(self):
        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["ING", "ABN"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        g = ge.DiGraph(self.v, e, v_id="name", src="creditor", dst="debtor")

        assert isinstance(g, ge.DiGraph)

        adj = self.adj != 0
        res = g.adjacency_matrix().todense()

        assert np.all(res == adj), res

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["RAB", "ABN"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        with pytest.raises(Exception) as e_info:
            ge.DiGraph(self.v, e, v_id="name", src="creditor", dst="debtor")

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

        e = pd.DataFrame(
            [
                ["ING", "ABN"],
                ["BNP", "ABN"],
                ["ING", "RAB"],
                ["ABN", "BNP"],
                ["BNP", "ING"],
            ],
            columns=["creditor", "debtor"],
        )

        with pytest.raises(Exception) as e_info:
            ge.DiGraph(self.v, e, v_id="name", src="creditor", dst="debtor")

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

    def test_vertex_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        test_v = np.array([1, 0, 1], dtype=np.uint8)

        assert isinstance(g, ge.DiGraph)
        assert np.all(g.groups == test_v), g.groups

    def test_num_edges(self):
        g = ge.DiGraph(self.v, self.e, v_id="name", src="creditor", dst="debtor")

        assert g.num_edges() == 4

    def test_degree_init(self):
        v = pd.DataFrame(
            [["ING"], ["ABN"], ["BNP"], ["RAB"], ["UBS"]], columns=["name"]
        )
        d = np.array([2, 2, 2, 0, 0])

        with pytest.warns(UserWarning):
            g = ge.DiGraph(v, self.e, v_id="name", src="creditor", dst="debtor")

        assert np.all(g.degree() == d), g.degree()

    def test_total_weight(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )

        assert g.total_weight() == 1e6 + 1.7e5 + 1e4 + 3e3

    def test_strength(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        s = np.array([1e6 + 1.7e5 + 1e4, 1.7e5 + 1e4 + 3e3, 1e6 + 3e3])
        s_test = g.strength()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength == s), g._strength

    def test_degree_by_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        d = np.array([[1, 1], [0, 2], [1, 1]])

        d_test = g.degree_by_group().todense()

        assert np.all(d_test == d), d_test
        assert np.all(g._degree_by_group.todense() == d), d_test

    def test_strength_by_group(self):
        g = ge.DiGraph(
            self.v,
            self.e,
            v_id="name",
            weight="value",
            src="creditor",
            dst="debtor",
            v_group="country",
        )

        s = np.array([(1.7e5 + 1e4, 1e6), (0, 1.7e5 + 1e4 + 3e3), (3e3, 1e6)])
        s_test = g.strength_by_group()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength_by_group == s), g._strength_by_group

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame([["ING"], ["ABN"], ["BNP"], ["RAB"]], columns=["name"])

        with pytest.warns(UserWarning, match="RAB vertex has no edges."):
            ge.DiGraph(v, self.e, v_id="name", src="creditor", dst="debtor")

        v = pd.DataFrame(
            [["ING"], ["ABN"], ["BNP"], ["RAB"], ["UBS"]], columns=["name"]
        )

        with pytest.warns(UserWarning, match=r" vertices have no edges."):
            ge.DiGraph(v, self.e, v_id="name", src="creditor", dst="debtor")

    def test_out_degree(self):
        g = ge.DiGraph(self.v, self.e, v_id="name", src="creditor", dst="debtor")
        d_out = np.array([1, 2, 1])
        d_test = g.out_degree()

        assert np.all(d_test == d_out), d_test
        assert np.all(g._out_degree == d_out), g._out_degree

    def test_in_degree(self):
        g = ge.DiGraph(self.v, self.e, v_id="name", src="creditor", dst="debtor")
        d_in = np.array([2, 1, 1])
        d_test = g.in_degree()

        assert np.all(d_test == d_in), d_test
        assert np.all(g._in_degree == d_in), g._in_degree

    def test_out_strength(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        s_out = np.array([1e4, 1.7e5 + 3e3, 1e6])
        s_test = g.out_strength()

        assert np.all(s_test == s_out), s_test
        assert np.all(g._out_strength == s_out), g._out_strength

    def test_in_strength(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        s_in = np.array([1e6 + 1.7e5, 1e4, 3e3])
        s_test = g.in_strength()

        assert np.all(s_test == s_in), s_test
        assert np.all(g._in_strength == s_in), g._in_strength

    def test_out_degree_by_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        d = np.array([[1, 0], [0, 2], [0, 1]])

        d_test = g.out_degree_by_group().todense()

        assert np.all(d_test == d), d_test
        assert np.all(g._out_degree_by_group.todense() == d), d_test

    def test_in_degree_by_group(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        d = np.array([[1, 1], [0, 1], [1, 0]])

        d_test = g.in_degree_by_group().todense()

        assert np.all(d_test == d), d_test
        assert np.all(g._in_degree_by_group.todense() == d), d_test

    def test_out_strength_by_group(self):
        g = ge.DiGraph(
            self.v,
            self.e,
            v_id="name",
            weight="value",
            src="creditor",
            dst="debtor",
            v_group="country",
        )

        s = np.array([(1e4, 0), (0, 1.7e5 + 3e3), (0, 1e6)])

        s_test = g.out_strength_by_group()

        assert np.all(s_test == s), s_test
        assert np.all(g._out_strength_by_group == s), g._out_strength_by_group

    def test_in_strength_by_group(self):
        g = ge.DiGraph(
            self.v,
            self.e,
            v_id="name",
            weight="value",
            src="creditor",
            dst="debtor",
            v_group="country",
        )

        s = np.array([(1.7e5, 1e6), (0, 1e4), (3e3, 0)])
        s_test = g.in_strength_by_group()

        assert np.all(s_test == s), s_test
        assert np.all(g._in_strength_by_group == s), g._in_strength_by_group

    def test_av_nn_prop_ones(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.ones(g.num_vertices)
        avnn = g.average_nn_property(prop)

        assert np.all(avnn == prop), avnn

    def test_av_nn_prop_zeros(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.zeros(g.num_vertices)
        avnn = g.average_nn_property(prop)

        assert np.all(avnn == 0), avnn

    def test_av_nn_prop_scale(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        prop = np.arange(g.num_vertices)
        avnn = g.average_nn_property(prop)
        test = np.array([1, 1, 0])

        assert np.all(avnn == test), avnn

    def test_av_nn_deg(self):
        g = ge.DiGraph(
            self.v, self.e, v_id="name", src="creditor", dst="debtor", weight="value"
        )
        avnn = g.average_nn_degree(ddir="out", ndir="out")
        test = np.array([2, 1, 1])
        assert np.all(avnn == test), avnn
        avnn = g.average_nn_degree(ddir="out", ndir="in")
        test = np.array([1.5, 1, 2])
        assert np.all(avnn == test), avnn
        avnn = g.average_nn_degree(ddir="in", ndir="out")
        test = np.array([1, 1.5, 2])
        assert np.all(avnn == test), avnn
        avnn = g.average_nn_degree(ddir="in", ndir="in")
        test = np.array([1, 2, 1])
        assert np.all(avnn == test), avnn
        avnn = g.average_nn_degree(ddir="out-in", ndir="out-in")
        test = np.array([2, 2, 2])
        assert np.all(avnn == test), avnn

    def test_multi_id_init(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["BNP", "IT"]],
            columns=["name", "country"],
        )

        e = pd.DataFrame(
            [
                ["ING", "NL", "ABN", "NL"],
                ["BNP", "FR", "ABN", "NL"],
                ["ABN", "NL", "BNP", "IT"],
                ["BNP", "IT", "ING", "NL"],
            ],
            columns=["creditor", "c_country", "debtor", "d_country"],
        )

        g = ge.DiGraph(
            v,
            e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
        )

        adj = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

        test_dict = {
            ("ING", "NL"): 3,
            ("ABN", "NL"): 0,
            ("BNP", "FR"): 1,
            ("BNP", "IT"): 2,
        }

        assert np.all(g.adjacency_matrix() == adj), g.adjacency_matrix()
        assert test_dict == g.id_dict

    def test_to_networkx(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v, self.e, v_id="name", src="creditor", dst="debtor", v_group="country"
        )

        v = [
            (0, {"node_id": "ABN", "group": "NL"}),
            (1, {"node_id": "BNP", "group": "FR"}),
            (2, {"node_id": "ING", "group": "NL"}),
        ]
        e = [
            (0, 1, {"weight": True}),
            (1, 0, {"weight": True}),
            (1, 2, {"weight": True}),
            (2, 0, {"weight": True}),
        ]
        adj = {
            0: {1: {"weight": True}},
            1: {0: {"weight": True}, 2: {"weight": True}},
            2: {0: {"weight": True}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj

    def test_to_networkx_weights(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"]], columns=["name", "country"]
        )

        g = ge.DiGraph(
            v,
            self.e,
            v_id="name",
            src="creditor",
            dst="debtor",
            v_group="country",
            weight="value",
        )

        v = [
            (0, {"node_id": "ABN", "group": "NL"}),
            (1, {"node_id": "BNP", "group": "FR"}),
            (2, {"node_id": "ING", "group": "NL"}),
        ]
        e = [
            (0, 1, {"weight": 1e4}),
            (1, 0, {"weight": 1.7e5}),
            (1, 2, {"weight": 3e3}),
            (2, 0, {"weight": 1e6}),
        ]
        adj = {
            0: {1: {"weight": 1e4}},
            1: {0: {"weight": 1.7e5}, 2: {"weight": 3e3}},
            2: {0: {"weight": 1e6}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v)
        assert np.all(list(gx.edges(data=True)) == e)
        assert gx.adj == adj


class TestMultiGraph:
    v = pd.DataFrame(
        [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["BNP", "IT"]],
        columns=["name", "country"],
    )

    e = pd.DataFrame(
        [
            ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
            ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
            ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
            ["BNP", "IT", "ABN", "NL", 3e3, "interbank", False],
            ["ABN", "NL", "BNP", "FR", 1e4, "interbank", False],
            ["ABN", "NL", "ING", "NL", 4e5, "external", True],
        ],
        columns=[
            "creditor",
            "c_country",
            "debtor",
            "d_country",
            "value",
            "type",
            "EUR",
        ],
    )
    v_s = pd.DataFrame(
        [["ING", "NL", 1e12], ["ABN", "NL", 5e11], ["BNP", "FR", 13e12]],
        columns=["name", "country", "assets"],
    )

    e_s = pd.DataFrame(
        [
            ["ING", "ABN", 1e6, "interbank"],
            ["BNP", "ABN", 2.3e7, "external"],
            ["BNP", "ABN", 1.7e5, "interbank"],
            ["ABN", "BNP", 1e4, "interbank"],
            ["ABN", "ING", 4e5, "external"],
        ],
        columns=["creditor", "debtor", "value", "type"],
    )

    adj = np.zeros((4, 4, 4), dtype=np.float64)
    adj[0, 1, 0] = 2.3e7
    adj[0, 0, 1] = 2.3e7
    adj[1, 0, 3] = 4e5
    adj[1, 3, 0] = 4e5
    adj[2, 0, 1] = 1e4
    adj[2, 1, 0] = 1e4
    adj[2, 0, 2] = 3e3
    adj[2, 2, 0] = 3e3
    adj[2, 0, 3] = 1e6
    adj[2, 3, 0] = 1e6
    adj[3, 0, 2] = 7e5
    adj[3, 2, 0] = 7e5

    adj_s = np.zeros((2, 3, 3), dtype=np.float64)
    adj_s[0, 0, 1] = 2.3e7
    adj_s[0, 1, 0] = 2.3e7
    adj_s[0, 0, 2] = 4e5
    adj_s[0, 2, 0] = 4e5
    adj_s[1, 0, 1] = 1e4 + 1.7e5
    adj_s[1, 1, 0] = 1e4 + 1.7e5
    adj_s[1, 0, 2] = 1e6
    adj_s[1, 2, 0] = 1e6

    def test_init_class(self):
        g = ge.MultiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )

        assert isinstance(g, ge.MultiGraph)
        assert g.num_edges() == 2

    def test_num_edges_by_label(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )
        test_num = np.array([1, 1, 3, 1])
        assert np.all(g.num_edges_label() == test_num)

    def test_init_edges(self):
        g = ge.MultiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )

        adj = self.adj_s.sum(axis=0) != 0
        adj_test = g.adjacency_matrix().todense()
        ten_test = np.array([list(x.todense()) for x in g.adjacency_tensor()])

        assert np.all(adj_test == adj), adj_test
        assert np.all(ten_test == (self.adj_s != 0)), ten_test

    def test_init_weights(self):
        g = ge.MultiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            weight="value",
            edge_label="type",
        )

        assert isinstance(g, ge.MultiGraph)
        adj = self.adj_s.sum(axis=0)
        adj_test = g.adjacency_matrix(weighted=True).todense()
        ten_test = np.array(
            [list(x.todense()) for x in g.adjacency_tensor(weighted=True)]
        )
        assert np.all(adj_test == adj), adj_test
        assert np.all(ten_test == self.adj_s), ten_test

    def test_init_id(self):
        g = ge.MultiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )
        test_dict = {"ABN": 0, "BNP": 1, "ING": 2}
        assert test_dict == g.id_dict

    def test_init_label(self):
        g = ge.MultiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )
        test_dict = {"external": 0, "interbank": 1}
        assert test_dict == g.label_dict

    def test_multi_id_init(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        test_dict = {
            ("ABN", "NL"): 0,
            ("BNP", "FR"): 1,
            ("BNP", "IT"): 2,
            ("ING", "NL"): 3,
        }
        test_label = {
            ("external", False): 0,
            ("external", True): 1,
            ("interbank", False): 2,
            ("interbank", True): 3,
        }

        adj = self.adj != 0
        ten_test = np.array([list(x.todense()) for x in g.adjacency_tensor()])
        wten_test = np.array(
            [list(x.todense()) for x in g.adjacency_tensor(weighted=True)]
        )
        assert np.all(ten_test == adj), ten_test
        assert np.all(wten_test == self.adj), wten_test
        assert test_dict == g.id_dict, g.id_dict
        assert test_label == g.label_dict, g.label_dict

    def test_duplicated_vertices(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["ABN", "NL"], ["BNP", "IT"]],
            columns=["name", "country"],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "There is at least one repeated id in the vertex dataframe."
        assert e_info.value.args[0] == msg

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame(
            [
                ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
                ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
                ["ABN", "UK", "BNP", "FR", 1e4, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 4e5, "external", False],
            ],
            columns=[
                "creditor",
                "c_country",
                "debtor",
                "d_country",
                "value",
                "type",
                "EUR",
            ],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiGraph(
                self.v,
                e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

        e = pd.DataFrame(
            [
                ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
                ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
                ["ABN", "NL", "UBS", "FR", 1e4, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 4e5, "external", False],
            ],
            columns=[
                "creditor",
                "c_country",
                "debtor",
                "d_country",
                "value",
                "type",
                "EUR",
            ],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiGraph(
                self.v,
                e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

    def test_vertex_group(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
        )

        test_v = np.array([2, 0, 1, 2], dtype=np.uint8)

        assert isinstance(g, ge.Graph)
        assert isinstance(g, ge.MultiGraph)
        assert np.all(g.groups == test_v), g.groups

    def test_degree(self):
        v = pd.DataFrame(
            [
                ["ING", "NL"],
                ["ABN", "NL"],
                ["BNP", "FR"],
                ["BNP", "IT"],
                ["ABN", "UK"],
                ["UBS", "FR"],
            ],
            columns=["name", "country"],
        )

        d = np.array([3, 0, 1, 1, 1, 0])

        with pytest.warns(UserWarning):
            g = ge.MultiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        assert np.all(g.degree() == d), g.degree()

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame(
            [
                ["ING", "NL"],
                ["ABN", "NL"],
                ["BNP", "FR"],
                ["BNP", "IT"],
                ["ABN", "UK"],
                ["UBS", "FR"],
            ],
            columns=["name", "country"],
        )

        with pytest.warns(UserWarning, match=r" vertices have no edges."):
            ge.MultiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

    def test_degree_by_label(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )

        d = np.array([[1, 1, 3, 1], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]])

        d_test = g.degree_by_label()

        assert np.all(d_test == d), d_test
        assert np.all(g._degree_by_label == d), g._degree_by_label

    def test_total_weight(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        assert g.total_weight() == 1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5

    def test_strength(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )
        s = np.array(
            [1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5, 2.3e7 + 1e4, 7e5 + 3e3, 1e6 + 4e5]
        )
        s_test = g.strength()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength == s), g._strength

    def test_total_weight_by_label(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        w_by_l = np.array([2.3e7, 4e5, 1e6 + 3e3 + 1e4, 7e5], dtype="f8")
        test = g.total_weight_label()
        assert np.all(test == w_by_l), test

    def test_strength_by_label(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        s = np.array(
            [
                [2.3e7, 4e5, 1e4 + 3e3 + 1e6, 7e5],
                [2.3e7, 0, 1e4, 0],
                [0, 0, 3e3, 7e5],
                [0, 4e5, 1e6, 0],
            ]
        )

        s_test = g.strength_by_label()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength_by_label == s), g._strength_by_label

    def test_to_networkx(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
        )

        v = [
            (0, {"node_id": ("ABN", "NL"), "group": "NL"}),
            (1, {"node_id": ("BNP", "FR"), "group": "FR"}),
            (2, {"node_id": ("BNP", "IT"), "group": "IT"}),
            (3, {"node_id": ("ING", "NL"), "group": "NL"}),
        ]

        e = [
            (0, 1, {"weight": True, "label": ("external", False)}),
            (0, 1, {"weight": True, "label": ("interbank", False)}),
            (0, 3, {"weight": True, "label": ("external", True)}),
            (0, 3, {"weight": True, "label": ("interbank", False)}),
            (0, 2, {"weight": True, "label": ("interbank", False)}),
            (0, 2, {"weight": True, "label": ("interbank", True)}),
        ]

        a = {
            0: {
                1: {
                    0: {"weight": True, "label": ("external", False)},
                    2: {"weight": True, "label": ("interbank", False)},
                },
                3: {
                    1: {"weight": True, "label": ("external", True)},
                    2: {"weight": True, "label": ("interbank", False)},
                },
                2: {
                    2: {"weight": True, "label": ("interbank", False)},
                    3: {"weight": True, "label": ("interbank", True)},
                },
            },
            1: {
                0: {
                    0: {"weight": True, "label": ("external", False)},
                    2: {"weight": True, "label": ("interbank", False)},
                }
            },
            2: {
                0: {
                    2: {"weight": True, "label": ("interbank", False)},
                    3: {"weight": True, "label": ("interbank", True)},
                }
            },
            3: {
                0: {
                    1: {"weight": True, "label": ("external", True)},
                    2: {"weight": True, "label": ("interbank", False)},
                }
            },
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v), gx.nodes(data=True)
        assert np.all(list(gx.edges(data=True)) == e), gx.edges(data=True)
        assert gx.adj == a, gx.adj

    def test_to_networkx_weights(self):
        g = ge.MultiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
            weight="value",
        )

        v = [
            (0, {"node_id": ("ABN", "NL"), "group": "NL"}),
            (1, {"node_id": ("BNP", "FR"), "group": "FR"}),
            (2, {"node_id": ("BNP", "IT"), "group": "IT"}),
            (3, {"node_id": ("ING", "NL"), "group": "NL"}),
        ]

        e = [
            (0, 1, {"weight": 2.3e7, "label": ("external", False)}),
            (0, 1, {"weight": 1e4, "label": ("interbank", False)}),
            (0, 3, {"weight": 4e5, "label": ("external", True)}),
            (0, 3, {"weight": 1e6, "label": ("interbank", False)}),
            (0, 2, {"weight": 3e3, "label": ("interbank", False)}),
            (0, 2, {"weight": 7e5, "label": ("interbank", True)}),
        ]

        a = {
            0: {
                1: {
                    0: {"weight": 23000000.0, "label": ("external", False)},
                    2: {"weight": 10000.0, "label": ("interbank", False)},
                },
                3: {
                    1: {"weight": 400000.0, "label": ("external", True)},
                    2: {"weight": 1000000.0, "label": ("interbank", False)},
                },
                2: {
                    2: {"weight": 3000.0, "label": ("interbank", False)},
                    3: {"weight": 700000.0, "label": ("interbank", True)},
                },
            },
            1: {
                0: {
                    0: {"weight": 23000000.0, "label": ("external", False)},
                    2: {"weight": 10000.0, "label": ("interbank", False)},
                }
            },
            2: {
                0: {
                    2: {"weight": 3000.0, "label": ("interbank", False)},
                    3: {"weight": 700000.0, "label": ("interbank", True)},
                }
            },
            3: {
                0: {
                    1: {"weight": 400000.0, "label": ("external", True)},
                    2: {"weight": 1000000.0, "label": ("interbank", False)},
                }
            },
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v), gx.nodes(data=True)
        assert np.all(list(gx.edges(data=True)) == e), gx.edges(data=True)
        assert gx.adj == a, gx.adj


class TestMultiDiGraph:
    v = pd.DataFrame(
        [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["BNP", "IT"]],
        columns=["name", "country"],
    )

    e = pd.DataFrame(
        [
            ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
            ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
            ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
            ["BNP", "IT", "ABN", "NL", 3e3, "interbank", False],
            ["ABN", "NL", "BNP", "FR", 1e4, "interbank", False],
            ["ABN", "NL", "ING", "NL", 4e5, "external", True],
        ],
        columns=[
            "creditor",
            "c_country",
            "debtor",
            "d_country",
            "value",
            "type",
            "EUR",
        ],
    )
    v_s = pd.DataFrame(
        [["ING", "NL", 1e12], ["ABN", "NL", 5e11], ["BNP", "FR", 13e12]],
        columns=["name", "country", "assets"],
    )

    e_s = pd.DataFrame(
        [
            ["ING", "ABN", 1e6, "interbank"],
            ["BNP", "ABN", 2.3e7, "external"],
            ["BNP", "ABN", 1.7e5, "interbank"],
            ["ABN", "BNP", 1e4, "interbank"],
            ["ABN", "ING", 4e5, "external"],
        ],
        columns=["creditor", "debtor", "value", "type"],
    )

    adj = np.zeros((4, 4, 4), dtype=np.float64)
    adj[0, 1, 0] = 2.3e7
    adj[1, 0, 3] = 4e5
    adj[2, 0, 1] = 1e4
    adj[2, 2, 0] = 3e3
    adj[2, 3, 0] = 1e6
    adj[3, 2, 0] = 7e5

    adj_s = np.zeros((2, 3, 3), dtype=np.float64)
    adj_s[0, 0, 2] = 4e5
    adj_s[0, 1, 0] = 2.3e7
    adj_s[1, 0, 1] = 1e4
    adj_s[1, 1, 0] = 1.7e5
    adj_s[1, 2, 0] = 1e6

    def test_init_class(self):
        g = ge.MultiDiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )

        assert isinstance(g, ge.MultiDiGraph)
        assert g.num_edges() == 4

    def test_num_edges_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )
        test_num = np.array([1, 1, 3, 1])
        assert np.all(g.num_edges_label() == test_num)

    def test_init_edges(self):
        g = ge.MultiDiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )

        adj = self.adj_s.sum(axis=0) != 0
        adj_test = g.adjacency_matrix().todense()
        ten_test = np.array([list(x.todense()) for x in g.adjacency_tensor()])

        assert np.all(adj_test == adj), adj_test
        assert np.all(ten_test == (self.adj_s != 0)), ten_test

    def test_init_weights(self):
        g = ge.MultiDiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            weight="value",
            edge_label="type",
        )

        assert isinstance(g, ge.MultiDiGraph)
        adj = self.adj_s.sum(axis=0)
        adj_test = g.adjacency_matrix(weighted=True).todense()
        ten_test = np.array(
            [list(x.todense()) for x in g.adjacency_tensor(weighted=True)]
        )
        assert np.all(adj_test == adj), adj_test
        assert np.all(ten_test == self.adj_s), ten_test

    def test_init_id(self):
        g = ge.MultiDiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )
        test_dict = {"ABN": 0, "BNP": 1, "ING": 2}
        assert test_dict == g.id_dict

    def test_init_label(self):
        g = ge.MultiDiGraph(
            self.v_s,
            self.e_s,
            v_id="name",
            src="creditor",
            dst="debtor",
            edge_label="type",
        )
        test_dict = {"external": 0, "interbank": 1}
        assert test_dict == g.label_dict

    def test_multi_id_init(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        test_dict = {
            ("ABN", "NL"): 0,
            ("BNP", "FR"): 1,
            ("BNP", "IT"): 2,
            ("ING", "NL"): 3,
        }
        test_label = {
            ("external", False): 0,
            ("external", True): 1,
            ("interbank", False): 2,
            ("interbank", True): 3,
        }

        adj = self.adj != 0
        ten_test = np.array([list(x.todense()) for x in g.adjacency_tensor()])
        wten_test = np.array(
            [list(x.todense()) for x in g.adjacency_tensor(weighted=True)]
        )
        assert np.all(ten_test == adj), ten_test
        assert np.all(wten_test == self.adj), wten_test
        assert test_dict == g.id_dict, g.id_dict
        assert test_label == g.label_dict, g.label_dict

    def test_duplicated_vertices(self):
        v = pd.DataFrame(
            [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["ABN", "NL"], ["BNP", "IT"]],
            columns=["name", "country"],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiDiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "There is at least one repeated id in the vertex dataframe."
        assert e_info.value.args[0] == msg

    def test_vertices_in_e_not_v(self):
        e = pd.DataFrame(
            [
                ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
                ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
                ["ABN", "UK", "BNP", "FR", 1e4, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 4e5, "external", False],
            ],
            columns=[
                "creditor",
                "c_country",
                "debtor",
                "d_country",
                "value",
                "type",
                "EUR",
            ],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiDiGraph(
                self.v,
                e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

        e = pd.DataFrame(
            [
                ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
                ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
                ["ABN", "NL", "UBS", "FR", 1e4, "interbank", False],
                ["BNP", "FR", "ABN", "NL", 4e5, "external", False],
            ],
            columns=[
                "creditor",
                "c_country",
                "debtor",
                "d_country",
                "value",
                "type",
                "EUR",
            ],
        )

        with pytest.raises(Exception) as e_info:
            ge.MultiDiGraph(
                self.v,
                e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        msg = "Some vertices in e are not in v."
        assert e_info.value.args[0] == msg

    def test_vertex_group(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
        )

        test_v = np.array([2, 0, 1, 2], dtype=np.uint8)

        assert isinstance(g, ge.Graph)
        assert isinstance(g, ge.MultiDiGraph)
        assert np.all(g.groups == test_v), g.groups

    def test_degree(self):
        v = pd.DataFrame(
            [
                ["ING", "NL"],
                ["ABN", "NL"],
                ["BNP", "FR"],
                ["BNP", "IT"],
                ["ABN", "UK"],
                ["UBS", "FR"],
            ],
            columns=["name", "country"],
        )

        d = np.array([3, 0, 1, 1, 1, 0])

        with pytest.warns(UserWarning):
            g = ge.MultiDiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

        assert np.all(g.degree() == d), g.degree()

    def test_out_degree(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )
        d_out = np.array([2, 1, 1, 1])
        d_test = g.out_degree()

        assert np.all(d_test == d_out), d_test
        assert np.all(g._out_degree == d_out), g._out_degree

    def test_in_degree(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )
        d_in = np.array([3, 1, 0, 1])
        d_test = g.in_degree()

        assert np.all(d_test == d_in), d_test
        assert np.all(g._in_degree == d_in), g._in_degree

    def test_vertices_with_no_edge(self):
        v = pd.DataFrame(
            [
                ["ING", "NL"],
                ["ABN", "NL"],
                ["BNP", "FR"],
                ["BNP", "IT"],
                ["ABN", "UK"],
                ["UBS", "FR"],
            ],
            columns=["name", "country"],
        )

        with pytest.warns(UserWarning, match=r" vertices have no edges."):
            ge.MultiDiGraph(
                v,
                self.e,
                v_id=["name", "country"],
                src=["creditor", "c_country"],
                dst=["debtor", "d_country"],
                edge_label=["type", "EUR"],
            )

    def test_degree_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )

        d = np.array([[1, 1, 3, 1], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]])

        d_test = g.degree_by_label()

        assert np.all(d_test == d), d_test
        assert np.all(g._degree_by_label == d), g._degree_by_label

    def test_out_degree_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )

        d_out = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]])

        d_test = g.out_degree_by_label()

        assert np.all(d_test == d_out), d_test
        assert np.all(g._out_degree_by_label == d_out), g._out_degree_by_label

    def test_in_degree_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
        )

        d_in = np.array([[1, 0, 2, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]])

        d_test = g.in_degree_by_label()

        assert np.all(d_test == d_in), d_test
        assert np.all(g._in_degree_by_label == d_in), g._in_degree_by_label

    def test_total_weight(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        assert g.total_weight() == 1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5

    def test_strength(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )
        s = np.array(
            [1e6 + 2.3e7 + 7e5 + 3e3 + 1e4 + 4e5, 2.3e7 + 1e4, 7e5 + 3e3, 1e6 + 4e5]
        )
        s_test = g.strength()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength == s), g._strength

    def test_out_strength(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )
        s_out = np.array([1e4 + 4e5, 2.3e7, 7e5 + 3e3, 1e6])
        s_test = g.out_strength()

        assert np.all(s_test == s_out), s_test
        assert np.all(g._out_strength == s_out), g._out_strength

    def test_in_strength(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )
        s_in = np.array([1e6 + 2.3e7 + 7e5 + 3e3, 1e4, 0, 4e5])
        s_test = g.in_strength()

        assert np.all(s_test == s_in), s_test
        assert np.all(g._in_strength == s_in), g._in_strength

    def test_total_weight_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        w_by_l = np.array([2.3e7, 4e5, 1e6 + 3e3 + 1e4, 7e5], dtype="f8")
        test = g.total_weight_label()
        assert np.all(test == w_by_l), test

    def test_strength_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        s = np.array(
            [
                [2.3e7, 4e5, 1e4 + 3e3 + 1e6, 7e5],
                [2.3e7, 0, 1e4, 0],
                [0, 0, 3e3, 7e5],
                [0, 4e5, 1e6, 0],
            ]
        )

        s_test = g.strength_by_label()

        assert np.all(s_test == s), s_test
        assert np.all(g._strength_by_label == s), g._strength_by_label

    def test_out_strength_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        s = np.array(
            [[0, 4e5, 1e4, 0], [2.3e7, 0, 0, 0], [0, 0, 3e3, 7e5], [0, 0, 1e6, 0]]
        )

        s_test = g.out_strength_by_label()

        assert np.all(s_test == s), s_test
        assert np.all(g._out_strength_by_label == s), s_test

    def test_in_strength_by_label(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            weight="value",
        )

        s = np.array(
            [[2.3e7, 0, 3e3 + 1e6, 7e5], [0, 0, 1e4, 0], [0, 0, 0, 0], [0, 4e5, 0, 0]]
        )

        s_test = g.in_strength_by_label()

        assert np.all(s_test == s), s_test
        assert np.all(g._in_strength_by_label == s), s_test

    def test_to_networkx(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
        )

        v = [
            (0, {"node_id": ("ABN", "NL"), "group": "NL"}),
            (1, {"node_id": ("BNP", "FR"), "group": "FR"}),
            (2, {"node_id": ("BNP", "IT"), "group": "IT"}),
            (3, {"node_id": ("ING", "NL"), "group": "NL"}),
        ]

        e = [
            (0, 3, {"weight": True, "label": ("external", True)}),
            (0, 1, {"weight": True, "label": ("interbank", False)}),
            (1, 0, {"weight": True, "label": ("external", False)}),
            (2, 0, {"weight": True, "label": ("interbank", False)}),
            (2, 0, {"weight": True, "label": ("interbank", True)}),
            (3, 0, {"weight": True, "label": ("interbank", False)}),
        ]

        a = {
            0: {
                1: {2: {"weight": True, "label": ("interbank", False)}},
                3: {1: {"weight": True, "label": ("external", True)}},
            },
            1: {0: {0: {"weight": True, "label": ("external", False)}}},
            2: {
                0: {
                    2: {"weight": True, "label": ("interbank", False)},
                    3: {"weight": True, "label": ("interbank", True)},
                }
            },
            3: {0: {2: {"weight": True, "label": ("interbank", False)}}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v), gx.nodes(data=True)
        assert np.all(list(gx.edges(data=True)) == e), gx.edges(data=True)
        assert gx.adj == a, gx.adj

    def test_to_networkx_weights(self):
        g = ge.MultiDiGraph(
            self.v,
            self.e,
            v_id=["name", "country"],
            src=["creditor", "c_country"],
            dst=["debtor", "d_country"],
            edge_label=["type", "EUR"],
            v_group="country",
            weight="value",
        )

        v = [
            (0, {"node_id": ("ABN", "NL"), "group": "NL"}),
            (1, {"node_id": ("BNP", "FR"), "group": "FR"}),
            (2, {"node_id": ("BNP", "IT"), "group": "IT"}),
            (3, {"node_id": ("ING", "NL"), "group": "NL"}),
        ]

        e = [
            (0, 3, {"weight": 4e5, "label": ("external", True)}),
            (0, 1, {"weight": 1e4, "label": ("interbank", False)}),
            (1, 0, {"weight": 2.3e7, "label": ("external", False)}),
            (2, 0, {"weight": 3e3, "label": ("interbank", False)}),
            (2, 0, {"weight": 7e5, "label": ("interbank", True)}),
            (3, 0, {"weight": 1e6, "label": ("interbank", False)}),
        ]

        a = {
            0: {
                1: {2: {"weight": 10000.0, "label": ("interbank", False)}},
                3: {1: {"weight": 400000.0, "label": ("external", True)}},
            },
            1: {0: {0: {"weight": 23000000.0, "label": ("external", False)}}},
            2: {
                0: {
                    2: {"weight": 3000.0, "label": ("interbank", False)},
                    3: {"weight": 700000.0, "label": ("interbank", True)},
                }
            },
            3: {0: {2: {"weight": 1000000.0, "label": ("interbank", False)}}},
        }

        gx = g.to_networkx()
        assert np.all(list(gx.nodes(data=True)) == v), gx.nodes(data=True)
        assert np.all(list(gx.edges(data=True)) == e), gx.edges(data=True)
        assert gx.adj == a, gx.adj
