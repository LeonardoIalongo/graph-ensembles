""" Test the fitness model class on simple sample graph. """
import graph_ensembles.spark as ge
import numpy as np
import pandas as pd
import pytest
import re
from pyspark import SparkContext


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

g_l = ge.MultiDiGraph(
    v,
    e_l,
    v_id=["name", "country"],
    src=["creditor", "c_country"],
    dst=["debtor", "d_country"],
    edge_label=["type", "EUR"],
    weight="value",
)

# Define graph marginals to check computation
out_strength = np.array([1e4 + 4e5, 2.3e7, 7e5 + 3e3, 1e6], dtype=np.float64)

in_strength = np.array([1e6 + 3e3 + 2.3e7 + 7e5, 1e4, 0, 4e5], dtype=np.float64)

num_vertices = 4
num_edges = 5

z = 4.291803e-12
p_ref = np.array(
    [
        [0.00000000, 0.01729211, 0.00000000, 0.41309584],
        [0.99959007, 0.00000000, 0.00000000, 0.97529924],
        [0.98676064, 0.02928772, 0.00000000, 0.54686647],
        [0.99065599, 0.04115187, 0.00000000, 0.00000000],
    ]
)

z_self = 5.27016837e-13
p_self = np.array(
    [
        [0.84221524, 0.00215611, 0.00000000, 0.07955478],
        [0.99667149, 0.10810950, 0.00000000, 0.82901759],
        [0.90150000, 0.00369125, 0.00000000, 0.12906942],
        [0.92866771, 0.00524254, 0.00000000, 0.17410436],
    ]
)

# Initialize spark
sc = SparkContext.getOrCreate()


class TestFitnessModelInit:
    def test_issubclass(self):
        """Check that the model is a graph ensemble."""
        model = ge.FitnessModel(sc, g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.FitnessModel(sc, g)
        assert np.all(model.prop_out == out_strength), g.out_strength()
        assert np.all(model.prop_in == in_strength), g.in_strength()
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

        model = ge.FitnessModel(sc, g)
        assert np.all(model.prop_out == out_strength)
        assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
        """Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            num_edges=num_edges,
        )
        assert np.all(model.prop_out == out_strength)
        assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_z(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z,
        )
        assert np.all(model.prop_out == out_strength)
        assert np.all(model.prop_in == in_strength)
        assert np.all(model.param == z)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

    def test_model_init_z_selfloops(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_self,
            selfloops=True,
        )
        assert np.all(model.prop_out == out_strength)
        assert np.all(model.prop_in == in_strength)
        assert np.all(model.param == z_self)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

    def test_model_wrong_init(self):
        """Check that the model raises exceptions for wrong inputs."""
        msg = "First argument must be a SparkContext."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(g, "df", 234, out_strength)

        msg = "A SparkContext must be passed as the first argument."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(prop_out=out_strength, prop_in=in_strength)

        msg = "Second argument passed must be a DiGraph."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(sc, "df", 234, out_strength)

        msg = "Unnamed arguments other than the Graph have been ignored."
        with pytest.warns(UserWarning, match=msg):
            ge.FitnessModel(sc, g, "df", 234, out_strength)

        msg = "Illegal argument passed: num_nodes"
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_nodes=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

    def test_wrong_num_vertices(self):
        """Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = "Number of vertices not set."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc, prop_out=out_strength, prop_in=in_strength, num_edges=num_edges
            )

        msg = "Number of vertices must be an integer."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=np.array([1, 2]),
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Number of vertices must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=-3,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

    def test_wrong_strengths(self):
        """Check that wrong initialization of strengths results in an error."""
        msg = "prop_out not set."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc, num_vertices=num_vertices, prop_in=in_strength, num_edges=num_edges
            )

        msg = "prop_in not set."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                num_edges=num_edges,
            )

        msg = "Node out properties must be a numpy array of length " + str(num_vertices)
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=1,
                prop_in=in_strength,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength[0:2],
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Node in properties must be a numpy array of length " + str(num_vertices)
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=2,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength[0:2],
                num_edges=num_edges,
            )

        msg = "Node out properties must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=-out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Node in properties must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=-in_strength,
                num_edges=num_edges,
            )

    def test_wrong_num_edges(self):
        """Check that wrong initialization of num_edges results in an error."""
        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=np.array([1, 2]),
            )

        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges="3",
            )

        msg = "Number of edges must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=-324,
            )

        msg = "Either num_edges or param must be set."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
            )

    def test_wrong_z(self):
        """Check that the passed z adheres to format."""
        msg = "The model requires one parameter."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param=np.array([0, 1]),
            )

        msg = "Parameters must be numeric."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param="0",
            )

        msg = "Parameters must be positive."
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(
                sc,
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param=-1,
            )


class TestFitnessModelFit:
    def test_solver_newton(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.FitnessModel(sc, g)
        model.fit(method="density", atol=1e-6)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """Check that it works with a given initial condition."""
        model = ge.FitnessModel(sc, g)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """Check that it works with a given initial condition."""
        model = ge.FitnessModel(sc, g)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_selfloops(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.FitnessModel(sc, g, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init_selfloops(self):
        """Check that it works with a given initial condition."""
        model = ge.FitnessModel(sc, g, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init_selfloops(self):
        """Check that it works with a given initial condition."""
        model = ge.FitnessModel(sc, g, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """Check that it raises an error with a negative initial condition."""
        model = ge.FitnessModel(sc, g)
        msg = "x0 must be positive."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=-1)

        msg = "The model requires one parameter."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0, 1]))

        msg = "x0 must be numeric."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0="hi")

    def test_wrong_method(self):
        """Check that wrong methods names return an error."""
        model = ge.FitnessModel(sc, g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")


class TestFitnessModelMeasures:
    model = ge.FitnessModel(
        sc,
        num_vertices=num_vertices,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z,
    )

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
        np.testing.assert_allclose(res, np.array([1, 1, 0, 1]), atol=1e-6, rtol=0)

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
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
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

    def test_confusion_matrix(self):
        """Test positive and negative counts."""
        adj = g.adjacency_matrix().todense()
        thresholds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        tp = np.zeros(thresholds.shape, dtype=int)
        fp = np.zeros(thresholds.shape, dtype=int)
        tn = np.zeros(thresholds.shape, dtype=int)
        fn = np.zeros(thresholds.shape, dtype=int)
        for i, th in enumerate(thresholds):
            tp[i] = np.sum(adj[p_ref >= th])
            fp[i] = np.sum(1 - adj[p_ref >= th])
            tn[i] = np.sum(1 - adj[p_ref < th])
            fn[i] = np.sum(adj[p_ref < th])
        # Remove selfloops
        fp[0] -= 4
        tn[1:] -= 4

        test = self.model.confusion_matrix(g, thresholds=thresholds)
        assert np.all(test[0] == tp), (test[0], tp)
        assert np.all(test[1] == fp), (test[1], fp)
        assert np.all(test[2] == tn), (test[2], tn)
        assert np.all(test[3] == fn), (test[3], fn)


class TestFitnessModelMeasuresSelfloops:
    model = ge.FitnessModel(
        sc,
        num_vertices=num_vertices,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z_self,
        selfloops=True,
    )

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
        np.testing.assert_allclose(res, np.array([1, 1, 0, 1]), atol=1e-6, rtol=0)

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
        p_u = 1 - (1 - p_self) * (1 - p_self.T)
        p_u.ravel()[:: num_vertices + 1] = 0
        d = p_u.sum(axis=0)
        d_out = p_self.sum(axis=1) - np.diagonal(p_self)
        d_in = p_self.sum(axis=0) - np.diagonal(p_self)

        exp = np.dot(p_self, prop) - np.diagonal(p_self) * prop
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_self.T, prop) - np.diagonal(p_self) * prop
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_u, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""
        d_out = self.model.expected_out_degree() - p_self.ravel()[:: num_vertices + 1]
        d_in = self.model.expected_in_degree() - p_self.ravel()[:: num_vertices + 1]

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(p_self, d_out) - np.diagonal(p_self) * d_out
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(p_self.T, d_out) - np.diagonal(p_self) * d_out
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(p_self.T, d_in) - np.diagonal(p_self) * d_in
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(p_self, d_in) - np.diagonal(p_self) * d_in
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree() - p_self.ravel()[:: num_vertices + 1]
        p_u = 1 - (1 - p_self) * (1 - p_self.T)
        p_u.ravel()[:: num_vertices + 1] = 0
        exp = np.dot(p_u, d)
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
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
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

    def test_confusion_matrix(self):
        """Test positive and negative counts."""
        adj = g.adjacency_matrix().todense()
        thresholds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        tp = np.zeros(thresholds.shape, dtype=int)
        fp = np.zeros(thresholds.shape, dtype=int)
        tn = np.zeros(thresholds.shape, dtype=int)
        fn = np.zeros(thresholds.shape, dtype=int)
        for i, th in enumerate(thresholds):
            tp[i] = np.sum(adj[p_self >= th])
            fp[i] = np.sum(1 - adj[p_self >= th])
            tn[i] = np.sum(1 - adj[p_self < th])
            fn[i] = np.sum(adj[p_self < th])

        test = self.model.confusion_matrix(g, thresholds=thresholds)
        assert np.all(test[0] == tp), (test[0], tp)
        assert np.all(test[1] == fp), (test[1], fp)
        assert np.all(test[2] == tn), (test[2], tn)
        assert np.all(test[3] == fn), (test[3], fn)


class TestFitnessModelSample:
    def test_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z,
        )

        samples = 20
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_selfloops(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.FitnessModel(
            sc,
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_self,
            selfloops=True,
        )

        samples = 20
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3
