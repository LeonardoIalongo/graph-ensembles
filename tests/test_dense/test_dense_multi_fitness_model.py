""" Test the fitness model class on simple sample graph. """
import graph_ensembles as ge
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
        ["ING", "NL", "ABN", "NL", 1e6, "interbank", False],
        ["BNP", "FR", "ABN", "NL", 2.3e7, "external", False],
        ["BNP", "IT", "ABN", "NL", 7e5, "interbank", True],
        ["BNP", "IT", "ABN", "NL", 3e3, "interbank", False],
        ["ABN", "NL", "BNP", "FR", 1e4, "interbank", False],
        ["ABN", "NL", "ING", "NL", 4e5, "external", True],
    ],
    columns=["creditor", "c_country", "debtor", "d_country", "value", "type", "EUR"],
)

g = ge.MultiDiGraph(
    v,
    e,
    v_id=["name", "country"],
    src=["creditor", "c_country"],
    dst=["debtor", "d_country"],
    edge_label=["type", "EUR"],
    weight="value",
)

# Define graph marginals to check computation
out_strength = np.array(
    [
        [0.0e00, 4.0e05, 1.0e04, 0.0e00],
        [2.3e07, 0.0e00, 0.0e00, 0.0e00],
        [0.0e00, 0.0e00, 3.0e03, 7.0e05],
        [0.0e00, 0.0e00, 1.0e06, 0.0e00],
    ]
)

in_strength = np.array(
    [
        [2.300e07, 0.000e00, 1.003e06, 7.000e05],
        [0.000e00, 0.000e00, 1.000e04, 0.000e00],
        [0.000e00, 0.000e00, 0.000e00, 0.000e00],
        [0.000e00, 4.000e05, 0.000e00, 0.000e00],
    ]
)

num_vertices = 4
num_labels = 4
num_edges = 5
num_edges_label = np.array([1, 1, 3, 1])

# Define p_ref for testing purposes (from z)
z = 8.98906315567293e-10
p_ref = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
i = np.array([0, 1, 2, 2, 2, 2, 2, 3])
j = np.array([1, 0, 0, 2, 2, 3, 3, 2])
k = np.array([0, 3, 1, 0, 1, 0, 1, 0])
v = np.array(
    [
        0.9999979,
        0.99309512,
        0.08247674,
        0.73008056,
        0.02625906,
        0.99889209,
        0.89989051,
        0.99773481,
    ]
)
p_ref[i, j, k] = v

# Define p_ref for testing purposes (from z_self)
z_self = 1.1263141e-10
p_self = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
i = np.array([0, 1, 2, 2, 2, 2, 2, 2, 3])
j = np.array([1, 0, 0, 0, 2, 2, 3, 3, 2])
k = np.array([0, 3, 0, 1, 0, 1, 0, 1, 0])
v = np.array(
    [
        0.99998322,
        0.9474266,
        0.53044876,
        0.0111377,
        0.25312265,
        0.00336756,
        0.99122571,
        0.5297026,
        0.98220305,
    ]
)
p_self[i, j, k] = v

# Define p_ref for testing purposes (from z_label)
z_label = np.array([2.02975770e-06, 6.71088639e-03, 1.82652908e-09, 2.19130984e-03])
p_ref_lbl = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
i = np.array([0, 1, 2, 2, 2, 2, 2, 3])
j = np.array([1, 0, 0, 2, 2, 3, 3, 2])
k = np.array([0, 3, 1, 0, 1, 0, 1, 0])
v = np.array(
    [1.0, 1.0, 0.15444338, 0.84605973, 0.05194927, 0.99945445, 0.94809318, 1.0]
)
p_ref_lbl[i, j, k] = v

# Define p_ref for testing purposes (from z_label)
z_lbl_self = np.array([2.02975770e-06, 6.71088639e-03, 2.95647709e-10, 2.19130984e-03])
p_self_lbl = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
i = np.array([0, 1, 2, 2, 2, 2, 2, 2, 3])
j = np.array([1, 0, 0, 0, 2, 2, 3, 3, 2])
k = np.array([0, 3, 0, 1, 0, 1, 0, 1, 0])
v = np.array(
    [
        1.0,
        1.0,
        0.74781523,
        0.02871579,
        0.47078858,
        0.00879146,
        0.99663905,
        0.74724989,
        1.0,
    ]
)
p_self_lbl[i, j, k] = v


class TestFitnessModelInit:
    def test_issubclass(self):
        """Check that the model is a graph ensemble."""
        model = ge.MultiFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.MultiFitnessModel(g, per_label=False)
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert not model.per_label

        model = ge.MultiFitnessModel(g, per_label=True)
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert model.per_label

    def test_model_init_param(self):
        """Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            num_edges=num_edges,
        )
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert not model.per_label

        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            num_edges_label=num_edges_label,
        )
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges_label == num_edges_label)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert model.per_label

    def test_model_init_z(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z,
        )
        assert np.all(model.param == z)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_label,
        )
        assert np.all(model.param == z_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        np.testing.assert_allclose(
            model.expected_num_edges_label(), num_edges_label, rtol=1e-5
        )

    def test_model_init_z_selfloops(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_self,
            selfloops=True,
        )
        assert np.all(model.param == z_self)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        assert model.selfloops
        np.testing.assert_allclose(model.expected_num_edges(), num_edges, rtol=1e-5)

        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_lbl_self,
            selfloops=True,
        )
        assert np.all(model.param == z_lbl_self)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        assert model.selfloops
        np.testing.assert_allclose(
            model.expected_num_edges_label(), num_edges_label, rtol=1e-5
        )

    def test_model_wrong_init(self):
        """Check that the model raises exceptions for wrong inputs."""
        msg = "First argument passed must be a MultiDiGraph."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel("df", 234, out_strength)

        msg = "Unnamed arguments other than the Graph have been ignored."
        with pytest.warns(UserWarning, match=msg):
            ge.MultiFitnessModel(g, "df", 234, out_strength)

        msg = "Illegal argument passed: num_nodes"
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
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
            ge.MultiFitnessModel(
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param=z_label,
            )

        msg = "Number of vertices must be an integer."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=np.array([1, 2]),
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param=z_label,
            )

        msg = "Number of vertices must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=-3,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param=z_label,
            )

    def test_wrong_num_labels(self):
        """Check that wrong initialization of num_labels results in an error."""
        msg = "Number of labels not set."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = re.escape(
            "Node out properties must be a two dimensional array "
            "with shape (num_vertices, num_labels)."
        )
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=2,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Number of labels must be an integer."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=np.array([1, 2]),
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Number of labels must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=-5,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

    def test_wrong_strengths(self):
        """Check that wrong initialization of strengths results in an error."""
        msg = "prop_out not set."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "prop_in not set."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                num_edges=num_edges,
            )

        msg = re.escape(
            "Node out properties must be a two dimensional array "
            "with shape (num_vertices, num_labels)."
        )
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=1,
                prop_in=in_strength,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength[0:2, 0:1],
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = re.escape(
            "Node in properties must be a two dimensional array "
            "with shape (num_vertices, num_labels)."
        )
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=3,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength[0:2, 0:1],
                num_edges=num_edges,
            )

        msg = "Node out properties must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=-out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Node in properties must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=-in_strength,
                num_edges=num_edges,
            )

    def test_wrong_num_edges(self):
        """Check that wrong initialization of num_edges results in an error."""
        msg = "num_edges must be an array of size one or num_labels, not 2."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=np.array([1, 2]),
            )

        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges="3",
            )

        msg = "Number of edges must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=-23,
            )

        msg = re.escape("Either num_edges(_label) or param must be set.")
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
            )

    def test_wrong_z(self):
        """Check that the passed z adheres to format."""
        msg = "The model requires one or num_labels parameter."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param=np.array([0, 1]),
            )

        msg = "Parameters must be numeric."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param="0",
            )

        msg = "Parameters must be positive."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(
                num_vertices=num_vertices,
                num_labels=num_labels,
                prop_out=out_strength,
                prop_in=in_strength,
                param=-1,
            )


class TestFitnessModelFit:
    def test_solver_newton(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_per_label(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_label, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init_per_label(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_label[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_bad_init_per_label(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_label[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """Check that it raises an error with a negative initial condition."""
        model = ge.MultiFitnessModel(g, per_label=True)
        msg = "x0 must be positive."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=-1)

        msg = "The model requires one or num_labels initial conditions."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0, 1]))

        msg = "The model requires one or num_labels initial conditions."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0="hi")

    def test_wrong_method(self):
        """Check that wrong methods names return an error."""
        model = ge.MultiFitnessModel(g, per_label=True)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")


class TestFitnessModelFitSelfloops:
    def test_solver_newton(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_per_label(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_init_per_label(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_bad_init_per_label(self):
        """Check that it works with a given initial condition."""
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)


class TestFitnessModelMeasures:
    model = ge.MultiFitnessModel(
        num_vertices=num_vertices,
        num_labels=num_labels,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z,
    )
    p = p_ref
    p_sym = 1 - (1 - p) * (1 - p.transpose((0, 2, 1)))
    diag_index = [
        i * (p_ref.shape[1] + 1) - p_ref.shape[1] * (i // p_ref.shape[1])
        for i in range(p_ref.shape[0] * p_ref.shape[1])
    ]
    p_sym.ravel()[diag_index] = np.diagonal(p_ref, axis1=1, axis2=2).ravel()
    p_proj = 1 - np.prod(1 - p, axis=0)
    p_proj_sym = 1 - (1 - p_proj) * (1 - p_proj.T)
    p_proj_sym.ravel()[:: num_vertices + 1] = p_proj.ravel()[:: p_proj.shape[1] + 1]

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    def test_exp_n_edges_label(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges_label()
        np.testing.assert_allclose(ne, self.p.sum(axis=(1, 2)), rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        np.testing.assert_allclose(d, self.p_proj_sym.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, self.p_proj.sum(axis=1), rtol=1e-5)

    def test_exp_in_degree(self):
        """Check expected d_out is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, self.p_proj.sum(axis=0), rtol=1e-5)

    def test_exp_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_degree_by_label()
        np.testing.assert_allclose(d, self.p_sym.sum(axis=2).T, rtol=1e-6)

    def test_exp_out_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_out_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=2).T, rtol=1e-6)

    def test_exp_in_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_in_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=1).T, rtol=1e-6)

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
        d = self.p_proj_sym.sum(axis=0)
        d_out = self.p_proj.sum(axis=1)
        d_in = self.p_proj.sum(axis=0)

        exp = np.dot(self.p_proj, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj_sym, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""

        d_out = self.model.expected_out_degree()
        d_in = self.model.expected_in_degree()

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(self.p_proj, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(self.p_proj.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(self.p_proj.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(self.p_proj, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree()
        exp = np.dot(self.p_proj_sym, d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood_2D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = self.p_proj.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p_proj != 1
        np_log[ind] = np.log1p(-self.p_proj[ind])
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_2D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
            selfloops=False,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_error(self):
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

    def test_likelihood_3D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = np.full(self.p.shape, -np.infty)
        ind = self.p > 0
        p_log[ind] = np.log(self.p[ind])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p < 1
        np_log[ind] = np.log1p(-self.p[ind])
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k]:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_tensor()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_3D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
            selfloops=False,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3, 0], [1, 0, 0, 2, 3, 2, 2], [0, 3, 1, 0, 0, 0, 3]] = True

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_error(self):
        """Test likelihood code."""
        adj = np.zeros((3, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 1], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        msg = re.escape(
            "Passed graph adjacency tensor does not have the "
            "correct shape: (3, 4, 4) instead of (4, 4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestFitnessModelMeasuresSelfloops:
    model = ge.MultiFitnessModel(
        num_vertices=num_vertices,
        num_labels=num_labels,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z_self,
        selfloops=True,
    )

    p = p_self
    p_sym = 1 - (1 - p) * (1 - p.transpose((0, 2, 1)))
    diag_index = [
        i * (p_self.shape[1] + 1) - p_self.shape[1] * (i // p_self.shape[1])
        for i in range(p_self.shape[0] * p_self.shape[1])
    ]
    p_sym.ravel()[diag_index] = np.diagonal(p_self, axis1=1, axis2=2).ravel()
    p_proj = 1 - np.prod(1 - p, axis=0)
    p_proj_sym = 1 - (1 - p_proj) * (1 - p_proj.T)
    p_proj_sym.ravel()[:: num_vertices + 1] = p_proj.ravel()[:: p_proj.shape[1] + 1]

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    def test_exp_n_edges_label(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges_label()
        np.testing.assert_allclose(ne, self.p.sum(axis=(1, 2)), rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        np.testing.assert_allclose(d, self.p_proj_sym.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, self.p_proj.sum(axis=1), rtol=1e-5)

    def test_exp_in_degree(self):
        """Check expected d_out is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, self.p_proj.sum(axis=0), rtol=1e-5)

    def test_exp_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_degree_by_label()
        np.testing.assert_allclose(d, self.p_sym.sum(axis=2).T, rtol=1e-6)

    def test_exp_out_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_out_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=2).T, rtol=1e-6)

    def test_exp_in_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_in_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=1).T, rtol=1e-6)

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
        d = self.p_proj_sym.sum(axis=0) - np.diagonal(self.p_proj_sym)
        d_out = self.p_proj.sum(axis=1) - np.diagonal(self.p_proj)
        d_in = self.p_proj.sum(axis=0) - np.diagonal(self.p_proj)

        exp = np.dot(self.p_proj, prop) - np.diagonal(self.p_proj) * prop
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj.T, prop) - np.diagonal(self.p_proj) * prop
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj_sym, prop) - np.diagonal(self.p_proj_sym) * prop
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""
        d_out = self.model.expected_out_degree() - np.diagonal(self.p_proj)
        d_in = self.model.expected_in_degree() - np.diagonal(self.p_proj)

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(self.p_proj, d_out) - np.diagonal(self.p_proj) * d_out
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(self.p_proj.T, d_out) - np.diagonal(self.p_proj) * d_out
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(self.p_proj.T, d_in) - np.diagonal(self.p_proj) * d_in
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(self.p_proj, d_in) - np.diagonal(self.p_proj) * d_in
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree() - np.diagonal(self.p_proj_sym)
        exp = np.dot(self.p_proj_sym, d) - np.diagonal(self.p_proj) * d
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood_2D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = self.p_proj.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p_proj != 1
        np_log[ind] = np.log1p(-self.p_proj[ind])
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_2D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
            selfloops=True,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_error(self):
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

    def test_likelihood_3D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = np.full(self.p.shape, -np.infty)
        ind = self.p > 0
        p_log[ind] = np.log(self.p[ind])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p < 1
        np_log[ind] = np.log1p(-self.p[ind])
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k]:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_tensor()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_3D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty]),
            selfloops=True,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3, 0], [1, 0, 0, 2, 3, 2, 2], [0, 3, 1, 0, 0, 0, 3]] = True

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_error(self):
        """Test likelihood code."""
        adj = np.zeros((3, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 1], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        msg = re.escape(
            "Passed graph adjacency tensor does not have the "
            "correct shape: (3, 4, 4) instead of (4, 4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestFitnessModelMeasuresPerlabel:
    model = ge.MultiFitnessModel(
        num_vertices=num_vertices,
        num_labels=num_labels,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z_label,
    )

    p = p_ref_lbl
    p_sym = 1 - (1 - p) * (1 - p.transpose((0, 2, 1)))
    diag_index = [
        i * (p_ref_lbl.shape[1] + 1) - p_ref_lbl.shape[1] * (i // p_ref_lbl.shape[1])
        for i in range(p_ref_lbl.shape[0] * p_ref_lbl.shape[1])
    ]
    p_sym.ravel()[diag_index] = np.diagonal(p_ref_lbl, axis1=1, axis2=2).ravel()
    p_proj = 1 - np.prod(1 - p, axis=0)
    p_proj_sym = 1 - (1 - p_proj) * (1 - p_proj.T)
    p_proj_sym.ravel()[:: num_vertices + 1] = p_proj.ravel()[:: p_proj.shape[1] + 1]

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        tmp = 1 - np.prod(1 - self.p, axis=0)
        np.testing.assert_allclose(ne, tmp.sum(), rtol=1e-5)

    def test_exp_n_edges_label(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges_label()
        np.testing.assert_allclose(ne, self.p.sum(axis=(1, 2)), rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        np.testing.assert_allclose(d, self.p_proj_sym.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, self.p_proj.sum(axis=1), rtol=1e-5)

    def test_exp_in_degree(self):
        """Check expected d_out is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, self.p_proj.sum(axis=0), rtol=1e-5)

    def test_exp_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_degree_by_label()
        np.testing.assert_allclose(d, self.p_sym.sum(axis=2).T, rtol=1e-6)

    def test_exp_out_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_out_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=2).T, rtol=1e-6)

    def test_exp_in_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_in_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=1).T, rtol=1e-6)

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
        d = self.p_proj_sym.sum(axis=0)
        d_out = self.p_proj.sum(axis=1)
        d_in = self.p_proj.sum(axis=0)

        exp = np.dot(self.p_proj, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj_sym, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""

        d_out = self.model.expected_out_degree()
        d_in = self.model.expected_in_degree()

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(self.p_proj, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(self.p_proj.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(self.p_proj.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(self.p_proj, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree()
        exp = np.dot(self.p_proj_sym, d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood_2D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = self.p_proj.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p_proj != 1
        np_log[ind] = np.log1p(-self.p_proj[ind])
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_2D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty] * num_labels),
            selfloops=False,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_error(self):
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

    def test_likelihood_3D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = np.full(self.p.shape, -np.infty)
        ind = self.p > 0
        p_log[ind] = np.log(self.p[ind])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p < 1
        np_log[ind] = np.log1p(-self.p[ind])
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k]:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_tensor()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_3D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty] * num_labels),
            selfloops=False,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3, 0], [1, 0, 0, 2, 3, 2, 2], [0, 3, 1, 0, 0, 0, 3]] = True

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_error(self):
        """Test likelihood code."""
        adj = np.zeros((3, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 1], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        msg = re.escape(
            "Passed graph adjacency tensor does not have the "
            "correct shape: (3, 4, 4) instead of (4, 4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestFitnessModelMeasuresSelfPerlabel:
    model = ge.MultiFitnessModel(
        num_vertices=num_vertices,
        num_labels=num_labels,
        prop_out=out_strength,
        prop_in=in_strength,
        param=z_lbl_self,
        selfloops=True,
    )

    p = p_self_lbl
    p_sym = 1 - (1 - p) * (1 - p.transpose((0, 2, 1)))
    diag_index = [
        i * (p_self_lbl.shape[1] + 1) - p_self_lbl.shape[1] * (i // p_self_lbl.shape[1])
        for i in range(p_self_lbl.shape[0] * p_self_lbl.shape[1])
    ]
    p_sym.ravel()[diag_index] = np.diagonal(p_self_lbl, axis1=1, axis2=2).ravel()
    p_proj = 1 - np.prod(1 - p, axis=0)
    p_proj_sym = 1 - (1 - p_proj) * (1 - p_proj.T)
    p_proj_sym.ravel()[:: num_vertices + 1] = p_proj.ravel()[:: p_proj.shape[1] + 1]

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        tmp = 1 - np.prod(1 - self.p, axis=0)
        np.testing.assert_allclose(ne, tmp.sum(), rtol=1e-5)

    def test_exp_n_edges_label(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges_label()
        np.testing.assert_allclose(ne, self.p.sum(axis=(1, 2)), rtol=1e-5)

    def test_exp_degree(self):
        """Check expected d is correct."""
        d = self.model.expected_degree()
        np.testing.assert_allclose(d, self.p_proj_sym.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """Check expected d_out is correct."""
        d_out = self.model.expected_out_degree()
        np.testing.assert_allclose(d_out, self.p_proj.sum(axis=1), rtol=1e-5)

    def test_exp_in_degree(self):
        """Check expected d_in is correct."""
        d_in = self.model.expected_in_degree()
        np.testing.assert_allclose(d_in, self.p_proj.sum(axis=0), rtol=1e-5)

    def test_exp_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_degree_by_label()
        np.testing.assert_allclose(d, self.p_sym.sum(axis=2).T, rtol=1e-6)

    def test_exp_out_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_out_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=2).T, rtol=1e-6)

    def test_exp_in_degree_by_label(self):
        """Check expected d_label is correct."""
        d = self.model.expected_in_degree_by_label()
        np.testing.assert_allclose(d, self.p.sum(axis=1).T, rtol=1e-6)

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
        d = self.p_proj_sym.sum(axis=0) - np.diagonal(self.p_proj_sym)
        d_out = self.p_proj.sum(axis=1) - np.diagonal(self.p_proj)
        d_in = self.p_proj.sum(axis=0) - np.diagonal(self.p_proj)

        exp = np.dot(self.p_proj, prop) - np.diagonal(self.p_proj) * prop
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj.T, prop) - np.diagonal(self.p_proj) * prop
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = self.model.expected_av_nn_property(prop, ndir="in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(self.p_proj_sym, prop) - np.diagonal(self.p_proj_sym) * prop
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = self.model.expected_av_nn_property(prop, ndir="out-in")
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """Test average nn degree."""
        d_out = self.model.expected_out_degree() - np.diagonal(self.p_proj)
        d_in = self.model.expected_in_degree() - np.diagonal(self.p_proj)

        self.model.expected_av_nn_degree(ddir="out", ndir="out")
        exp = np.dot(self.p_proj, d_out) - np.diagonal(self.p_proj) * d_out
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out", ndir="in")
        exp = np.dot(self.p_proj.T, d_out) - np.diagonal(self.p_proj) * d_out
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(
            self.model.exp_av_in_nn_d_out, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="in", ndir="in")
        exp = np.dot(self.p_proj.T, d_in) - np.diagonal(self.p_proj) * d_in
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp, atol=1e-5, rtol=0)

        self.model.expected_av_nn_degree(ddir="in", ndir="out")
        exp = np.dot(self.p_proj, d_in) - np.diagonal(self.p_proj) * d_in
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_nn_d_in, exp, atol=1e-5, rtol=0
        )

        self.model.expected_av_nn_degree(ddir="out-in", ndir="out-in")
        d = self.model.expected_degree() - np.diagonal(self.p_proj_sym)
        exp = np.dot(self.p_proj_sym, d) - np.diagonal(self.p_proj) * d
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(
            self.model.exp_av_out_in_nn_d_out_in, exp, atol=1e-5, rtol=0
        )

    def test_likelihood_2D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = self.p_proj.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p_proj != 1
        np_log[ind] = np.log1p(-self.p_proj[ind])
        adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_2D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty] * num_labels),
            selfloops=True,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_2D_error(self):
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

    def test_likelihood_3D(self):
        """Test likelihood code."""
        # Compute reference
        p_log = np.full(self.p.shape, -np.infty)
        ind = self.p > 0
        p_log[ind] = np.log(self.p[ind])
        np_log = np.full(p_log.shape, -np.infty)
        ind = self.p < 1
        np_log[ind] = np.log1p(-self.p[ind])
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k]:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(g.adjacency_tensor()), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
        )

    def test_likelihood_3D_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        # Construct model
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=np.array([np.infty] * num_labels),
            selfloops=True,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.zeros((4, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 3, 0], [1, 0, 0, 2, 3, 2, 2], [0, 3, 1, 0, 0, 0, 3]] = True

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_3D_error(self):
        """Test likelihood code."""
        adj = np.zeros((3, 4, 4), dtype=bool)
        adj[[0, 1, 2, 2, 2, 1], [1, 0, 0, 2, 3, 2], [0, 3, 1, 0, 0, 0]] = True

        msg = re.escape(
            "Passed graph adjacency tensor does not have the "
            "correct shape: (3, 4, 4) instead of (4, 4, 4)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestFitnessModelSample:
    def test_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z,
            selfloops=False,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_selfloops(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_self,
            selfloops=True,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_per_label(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_label,
            selfloops=False,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_selfloops_per_label(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.MultiFitnessModel(
            num_vertices=num_vertices,
            num_labels=num_labels,
            prop_out=out_strength,
            prop_in=in_strength,
            param=z_lbl_self,
            selfloops=True,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3
