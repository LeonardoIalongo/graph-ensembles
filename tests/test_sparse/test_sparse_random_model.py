""" Test the fitness model class on simple sample graph. """
import graph_ensembles.sparse as ge
import numpy as np
import pandas as pd
import pytest
import re

v = pd.DataFrame(
    [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["BNP", "IT"]],
    columns=["name", "country"],
)

e = pd.DataFrame(
    [
        ["ING", "NL", "ABN", "NL", 1e6],
        ["BNP", "FR", "ABN", "NL", 2.3e7],
        ["BNP", "IT", "ABN", "NL", 7e5 + 3e3],
        ["ABN", "NL", "BNP", "FR", 1e4],
        ["ABN", "NL", "ING", "NL", 4e5],
    ],
    columns=["creditor", "c_country", "debtor", "d_country", "value"],
)

e_l = pd.DataFrame(
    [
        ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
        ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
        ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
        ["BNP", "IT", "ABN", "NL", 3e3, "interbank", False],
        ["ABN", "NL", "BNP", "FR", 1e4, "interbank", False],
        ["ABN", "NL", "ING", "NL", 4e5, "external", True],
    ],
    columns=["creditor", "c_country", "debtor", "d_country", "value", "type", "EUR"],
)

g = ge.DiGraph(
    v,
    e,
    v_id=["name", "country"],
    src=["creditor", "c_country"],
    dst=["debtor", "d_country"],
    weight="value",
)

num_vertices = 4
num_edges = 5

z = 5 / 12
p_ref = np.empty((num_vertices, num_vertices))
p_ref[:] = z
p_ref.ravel()[:: num_vertices + 1] = 0

z_self = 5 / 16
p_self = np.empty((num_vertices, num_vertices))
p_self[:] = z_self


class TestRandomDiGraphInit:
    def test_issubclass(self):
        """Check that the model is a graph ensemble."""
        model = ge.RandomDiGraph(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.RandomDiGraph(g)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
        """Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.RandomDiGraph(num_vertices=num_vertices, num_edges=num_edges)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_z(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=z)
        assert np.all(model.param[0] == z)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

    def test_model_init_z_selfloops(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.RandomDiGraph(
            num_vertices=num_vertices, param=z_self, selfloops=True
        )
        assert np.all(model.param[0] == z_self)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

    def test_model_wrong_init(self):
        """Check that the model raises exceptions for wrong inputs."""
        msg = "First argument passed must be a Graph."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph("df", 234)

        msg = "Unnamed arguments other than the Graph have been ignored."
        with pytest.warns(UserWarning, match=msg):
            ge.RandomDiGraph(g, "df", 234, "sfd")

        msg = "Illegal argument passed: num_nodes"
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_nodes=num_vertices, num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = "Number of vertices not set."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_edges=num_edges)

        msg = "Number of vertices must be an integer."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=np.array([1, 2]), num_edges=num_edges)

        msg = "Number of vertices must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=-3, num_edges=num_edges)

    def test_wrong_num_edges(self):
        """Check that wrong initialization of num_edges results in an error."""
        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, num_edges=np.array([1, 2]))

        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, num_edges="3")

        msg = "Number of edges must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, num_edges=-324)

        msg = "Either num_edges or param must be set."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices)

    def test_wrong_z(self):
        """Check that the passed z adheres to format."""
        msg = "The model requires two parameter."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, param=np.array([0, 1, 3]))

        msg = "Parameters must be numeric."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, param="0")

        msg = "Parameters must be positive."
        with pytest.raises(ValueError, match=msg):
            ge.RandomDiGraph(num_vertices=num_vertices, param=-1)


class TestRandomDiGraphFit:
    def test_fit(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.RandomDiGraph(g)
        model.fit()
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param[0], atol=0, rtol=1e-6)

    def test_fit_selfloops(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.RandomDiGraph(g, selfloops=True)
        model.fit()
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param[0], atol=0, rtol=1e-6)


class TestRandomDiGraphMeasures:
    model = ge.RandomDiGraph(num_vertices=num_vertices, param=z)

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        d_ref = 1 - (1 - p_ref) * (1 - p_ref.T)  # Only valid if no self loops
        np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, p_ref.sum(axis=1), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_out))

    def test_exp_in_degree(self):
        """Check expected d_out is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, p_ref.sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_in))

    def test_av_nn_prop_ones(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.ones(num_vertices)
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, np.array([1, 1, 1, 1]), atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_zeros(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.zeros(num_vertices)
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_scale(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.arange(num_vertices) + 1
        p_u = 1 - (1 - p_ref) * (1 - p_ref.T)  # Only valid if no self loops
        d = p_u.sum(axis=0)
        d_out = p_ref.sum(axis=1)
        d_in = p_ref.sum(axis=0)

        exp = np.dot(p_ref, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_ref.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_u, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""
        d_out = self.model.expected_out_degree()
        d_in = self.model.expected_in_degree()

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(p_ref, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(p_ref.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(p_ref.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(p_ref, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree()
        exp = np.dot((1 - (1 - p_ref) * (1 - p_ref.T)), d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood(self):
        """Test likelihood code."""
        # Compute reference from p_ref
        p_log = p_ref.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.log1p(-p_ref)
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if i != j:
                    if adj[i, j] != 0:
                        ref += p_log[i, j]
                    else:
                        ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=np.array([1, 0]))

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=0)

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_error(self):
        """Test likelihood code."""
        adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])

        # Construct model
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=z)

        msg = re.escape(
            "Passed graph adjacency matrix does not have the "
            "correct shape: (3, 4) instead of (4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood("dfsg")


class TestRandomDiGraphMeasuresSelfloops:
    model = ge.RandomDiGraph(num_vertices=num_vertices, param=z_self, selfloops=True)

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        d_ref = 1 - (1 - p_self) * (1 - p_self.T)
        d_ref.ravel()[:: num_vertices + 1] = p_self.ravel()[:: num_vertices + 1]
        np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, p_self.sum(axis=1), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_out))

    def test_exp_in_degree(self):
        """Check expected d_out is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, p_self.sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_in))

    def test_av_nn_prop_ones(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.ones(num_vertices)
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, np.array([1, 1, 1, 1]), atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_zeros(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.zeros(num_vertices)
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_scale(self):
        """Test correct value of av_nn_prop using simple local prop."""
        prop = np.arange(num_vertices) + 1
        p_u = 1 - (1 - p_self) * (1 - p_self.T)  # Only valid if no self loops
        d = p_u.sum(axis=0) - (1 - (1 - z_self) ** 2)
        d_out = p_self.sum(axis=1) - z_self
        d_in = p_self.sum(axis=0) - z_self

        exp = np.dot(p_self, prop) - z_self * prop
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_self.T, prop) - z_self * prop
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_u, prop)
        exp -= (1 - (1 - z_self) ** 2) * prop
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""
        d_out = self.model.expected_out_degree() - z_self
        d_in = self.model.expected_in_degree() - z_self

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(p_self, d_out) - z_self * d_out
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(p_self.T, d_out) - z_self * d_out
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(p_self.T, d_in) - z_self * d_in
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(p_self, d_in) - z_self * d_in
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree() - z_self
        exp = np.dot((1 - (1 - p_self) * (1 - p_self.T)), d)
        exp -= (1 - (1 - z_self) ** 2) * d
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood(self):
        """Test likelihood code."""
        # Compute reference from p_self
        p_log = p_self.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.log1p(-p_self)
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=np.array([1, 0]))

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=0)

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_error(self):
        """Test likelihood code."""
        adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])

        msg = re.escape(
            "Passed graph adjacency matrix does not have the "
            "correct shape: (3, 4) instead of (4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestRandomDiGraphSample:
    def test_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.RandomDiGraph(num_vertices=num_vertices, param=z)

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_selfloops(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.RandomDiGraph(
            num_vertices=num_vertices, param=z_self, selfloops=True
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3
