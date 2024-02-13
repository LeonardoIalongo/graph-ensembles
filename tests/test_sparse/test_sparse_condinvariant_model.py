""" Test the fitness model class on simple sample graph. """
import graph_ensembles.sparse as ge
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pytest
import re

v = pd.DataFrame(
    [["ING", "NL"], ["ABN", "NL"], ["BNP", "FR"], ["BNP", "IT"], ["ING", "IT"]],
    columns=["name", "country"],
)

e = pd.DataFrame(
    [
        ["ING", "NL", "ABN", "NL", 1e6],
        ["BNP", "FR", "ABN", "NL", 2.3e7],
        ["BNP", "FR", "ING", "IT", 3.7e6],
        ["ING", "IT", "ING", "NL", 7e5],
        ["BNP", "IT", "ABN", "NL", 7e5 + 3e3],
        ["ABN", "NL", "BNP", "FR", 1e4],
        ["ABN", "NL", "ING", "NL", 4e5],
    ],
    columns=["creditor", "c_country", "debtor", "d_country", "value"],
)

g = ge.DiGraph(
    v,
    e,
    v_id=["name", "country"],
    src=["creditor", "c_country"],
    dst=["debtor", "d_country"],
    weight="value",
    v_group="country",
)

# Give aggregate matrix
gmat = np.zeros((g.num_groups, g.num_vertices), dtype="i4")
gmat[g.groups, np.arange(g.num_vertices)] = 1
gmat = sp.csr_array(gmat)
agg_adj = gmat.dot(g.adjacency_matrix(directed=True, weighted=True)).dot(gmat.T)

# Define graph marginals to check computation
out_strength = np.array([410000, 26700000, 703000, 700000, 1000000], dtype=np.float64)

in_strength = np.array([2.4703e7, 1.0e04, 0.0, 3.7e6, 1.1e6], dtype=np.float64)

num_vertices = 5
num_groups = 3
num_edges = 7

z = 1.240346e-13
p_ref = np.array(
    [
        [0.00000000, 0.01187471, 0.00000000, 0.37987302],
        [1.00000000, 0.00000000, 0.00000000, 1.00000000],
        [1.00000000, 0.02027429, 0.00000000, 0.55926231],
        [1.00000000, 0.02871568, 0.00000000, 0.00000000],
    ]
)

z_self = 6.623465e-14
p_self = np.array(
    [
        [9.01468767e-01, 9.37657399e-04, 0.00000000e00, 3.68285937e-02],
        [1.00000000e00, 5.12642490e-02, 0.00000000e00, 8.78154175e-01],
        [9.81191790e-01, 1.60720069e-03, 0.00000000e00, 6.23136286e-02],
        [9.96490039e-01, 2.28542656e-03, 0.00000000e00, 8.74584728e-02],
    ]
)


class TestInvariantModelInit:
    def test_issubclass(self):
        """Check that the model is a graph ensemble."""
        model = ge.ConditionalInvariantModel(g, agg_adj)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.ConditionalInvariantModel(g, agg_adj)
        assert np.all(model.prop_out == out_strength), g.out_strength()
        assert np.all(model.prop_in == in_strength), g.in_strength()
        assert np.all(model.groups == g.groups), g.groups
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_groups == num_groups)

    def test_model_init_param(self):
        """Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            num_edges=num_edges,
            groups=g.groups,
            adj=agg_adj,
        )
        assert np.all(model.prop_out == out_strength), g.out_strength()
        assert np.all(model.prop_in == in_strength), g.in_strength()
        assert np.all(model.groups == g.groups), g.groups
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_groups == num_groups)

    def test_model_init_z(self):
        """Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z,
            selfloops=False,
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
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
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
        msg = "First argument passed must be a DiGraph."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel("df", 234, out_strength)

        msg = "Second argument passed must be a DiGraph or an adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(g, "df", 234, out_strength)

        msg = "Unnamed arguments other than the Graph have been ignored."
        with pytest.warns(UserWarning, match=msg):
            ge.ConditionalInvariantModel(g, agg_adj, "df", 234, out_strength)

        msg = "Illegal argument passed: num_nodes"
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
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
            ge.ConditionalInvariantModel(
                prop_out=out_strength, prop_in=in_strength, num_edges=num_edges
            )

        msg = "Number of vertices must be an integer."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=np.array([1, 2]),
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Number of vertices must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=-3,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

    def test_wrong_strengths(self):
        """Check that wrong initialization of strengths results in an error."""
        msg = "prop_out not set."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices, prop_in=in_strength, num_edges=num_edges
            )

        msg = "prop_in not set."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices, prop_out=out_strength, num_edges=num_edges
            )

        msg = "Node out properties must be a numpy array of length " + str(num_vertices)
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=1,
                prop_in=in_strength,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength[0:2],
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Node in properties must be a numpy array of length " + str(num_vertices)
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=2,
                num_edges=num_edges,
            )
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength[0:2],
                num_edges=num_edges,
            )

        msg = "Node out properties must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=-out_strength,
                prop_in=in_strength,
                num_edges=num_edges,
            )

        msg = "Node in properties must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=-in_strength,
                num_edges=num_edges,
            )

    def test_wrong_num_edges(self):
        """Check that wrong initialization of num_edges results in an error."""
        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=np.array([1, 2]),
            )

        msg = "Number of edges must be a number."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges="3",
            )

        msg = "Number of edges must be a positive number."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                num_edges=-324,
            )

        msg = "Either num_edges or param must be set."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices, prop_out=out_strength, prop_in=in_strength
            )

    def test_wrong_z(self):
        """Check that the passed z adheres to format."""
        msg = "The model requires one parameter."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param=np.array([0, 1]),
            )

        msg = "Parameters must be numeric."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param="0",
            )

        msg = "Parameters must be positive."
        with pytest.raises(ValueError, match=msg):
            ge.ConditionalInvariantModel(
                num_vertices=num_vertices,
                prop_out=out_strength,
                prop_in=in_strength,
                param=-1,
            )


class TestInvariantModelFit:
    def test_solver_newton(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.ConditionalInvariantModel(g, agg_adj)
        model.fit(method="density", atol=1e-6)
        model.expected_num_edges()
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """Check that it works with a given initial condition."""
        model = ge.ConditionalInvariantModel(g, agg_adj)
        model.fit(x0=1e-14)
        model.expected_num_edges()
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """Check that it works with a given initial condition."""
        model = ge.ConditionalInvariantModel(g, agg_adj)
        model.fit(x0=1e2)
        model.expected_num_edges()
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_selfloops(self):
        """Check that the newton solver is fitting the z parameters
        correctly."""
        model = ge.ConditionalInvariantModel(g, agg_adj, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init_selfloops(self):
        """Check that it works with a given initial condition."""
        model = ge.ConditionalInvariantModel(g, agg_adj, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init_selfloops(self):
        """Check that it works with a given initial condition."""
        model = ge.ConditionalInvariantModel(g, agg_adj, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges, model.expected_num_edges(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """Check that it raises an error with a negative initial condition."""
        model = ge.ConditionalInvariantModel(g, agg_adj)
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
        model = ge.ConditionalInvariantModel(g, agg_adj)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")


class TestInvariantModelMeasures:
    model = ge.ConditionalInvariantModel(
        num_vertices=num_vertices,
        prop_out=out_strength,
        prop_in=in_strength,
        groups=g.groups,
        adj=agg_adj,
        param=z,
        selfloops=False,
    )

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    # def test_exp_degree(self):
    #     """Check expected d is correct."""
    #     d = self.model.expected_degree()
    #     d_ref = 1 - (1 - p_ref) * (1 - p_ref.T)  # Only valid if no self loops
    #     np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

    # def test_exp_out_degree(self):
    #     """Check expected d_out is correct."""
    #     d_out = self.model.expected_out_degree()
    #     np.testing.assert_allclose(d_out, p_ref.sum(axis=1), rtol=1e-5)
    #     np.testing.assert_allclose(num_edges, np.sum(d_out))

    # def test_exp_in_degree(self):
    #     """Check expected d_out is correct."""
    #     d_in = self.model.expected_in_degree()
    #     np.testing.assert_allclose(d_in, p_ref.sum(axis=0), rtol=1e-5)
    #     np.testing.assert_allclose(num_edges, np.sum(d_in))

    # def test_likelihood(self):
    #     """Test likelihood code."""
    #     # Compute reference from p_ref
    #     p_log = p_ref.copy()
    #     p_log[p_log != 0] = np.log(p_log[p_log != 0])
    #     np_log = -z * np.outer(out_strength, in_strength)
    #     adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

    #     ref = 0
    #     for i in range(adj.shape[0]):
    #         for j in range(adj.shape[1]):
    #             if i != j:
    #                 if adj[i, j] != 0:
    #                     ref += p_log[i, j]
    #                 else:
    #                     ref += np_log[i, j]

    #     # Construct model
    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
    #     )
    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
    #     )
    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
    #     )

    def test_likelihood_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ]
        )

        # Construct model
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=np.infty,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ]
        )

        # Construct model
        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_error(self):
        """Test likelihood code."""
        adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])

        # Construct model
        msg = re.escape(
            "Passed graph adjacency matrix does not have the "
            "correct shape: (3, 4) instead of (5, 5)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestInvariantModelMeasuresSelfloops:
    model = ge.ConditionalInvariantModel(
        num_vertices=num_vertices,
        prop_out=out_strength,
        prop_in=in_strength,
        groups=g.groups,
        adj=agg_adj,
        param=z_self,
        selfloops=True,
    )

    def test_exp_n_edges(self):
        """Check expected edges is correct."""
        ne = self.model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    # def test_exp_degree(self):
    #     """Check expected d is correct."""

    #     d = self.model.expected_degree()
    #     d_ref = 1 - (1 - p_self) * (1 - p_self.T)
    #     d_ref.ravel()[:: num_vertices + 1] = p_self.ravel()[:: num_vertices + 1]
    #     np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

    # def test_exp_out_degree(self):
    #     """Check expected d_out is correct."""

    #     d_out = self.model.expected_out_degree()
    #     np.testing.assert_allclose(d_out, p_self.sum(axis=1), rtol=1e-5)
    #     np.testing.assert_allclose(num_edges, np.sum(d_out))

    # def test_exp_in_degree(self):
    #     """Check expected d_out is correct."""

    #     d_in = self.model.expected_in_degree()
    #     np.testing.assert_allclose(d_in, p_self.sum(axis=0), rtol=1e-5)
    #     np.testing.assert_allclose(num_edges, np.sum(d_in))

    # def test_likelihood(self):
    #     """Test likelihood code."""
    #     # Compute reference from p_self
    #     p_log = p_self.copy()
    #     p_log[p_log != 0] = np.log(p_log[p_log != 0])
    #     np_log = -z_self * np.outer(out_strength, in_strength)
    #     adj = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

    #     ref = 0
    #     for i in range(adj.shape[0]):
    #         for j in range(adj.shape[1]):
    #             if adj[i, j] != 0:
    #                 ref += p_log[i, j]
    #             else:
    #                 ref += np_log[i, j]

    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(g), atol=1e-6, rtol=1e-6
    #     )
    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(g.adjacency_matrix()), atol=1e-6, rtol=1e-6
    #     )
    #     np.testing.assert_allclose(
    #         ref, self.model.log_likelihood(adj), atol=1e-6, rtol=1e-6
    #     )

    def test_likelihood_inf_p_one(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ]
        )

        # Construct model
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=np.infty,
            selfloops=True,
        )

        res = model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_inf_p_zero(self):
        """Test likelihood code."""
        # Construct adj with p[g] = 0
        adj = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ]
        )

        res = self.model.log_likelihood(adj)
        assert np.isinf(res) and (res < 0)

    def test_likelihood_error(self):
        """Test likelihood code."""
        adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])

        msg = re.escape(
            "Passed graph adjacency matrix does not have the "
            "correct shape: (3, 4) instead of (5, 5)"
        )
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood(adj)

        msg = "g input not a graph or adjacency matrix."
        with pytest.raises(ValueError, match=msg):
            self.model.log_likelihood("dfsg")


class TestInvariantModelSample:
    def test_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z,
            selfloops=False,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.all((agg_test.todense() != 0) == (agg_adj.todense() != 0))
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_cremb_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z,
            selfloops=False,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample(weights="cremb")
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.all((agg_test.todense() != 0) == (agg_adj.todense() != 0))
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_wagg_sampling(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z,
            selfloops=False,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample(weights="wagg")
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.abs((agg_test - agg_adj).todense()).max() < 1e-6
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_sampling_selfloops(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z_self,
            selfloops=True,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample()
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.all((agg_test.todense() != 0) == (agg_adj.todense() != 0))
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_cremb_sampling_self(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z_self,
            selfloops=True,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample(weights="cremb")
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.all((agg_test.todense() != 0) == (agg_adj.todense() != 0))
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3

    def test_wagg_sampling_self(self):
        """Check that properties of the sample correspond to ensemble."""
        model = ge.ConditionalInvariantModel(
            num_vertices=num_vertices,
            prop_out=out_strength,
            prop_in=in_strength,
            groups=g.groups,
            adj=agg_adj,
            param=z_self,
            selfloops=True,
        )

        samples = 100
        for i in range(samples):
            sample = model.sample(weights="wagg")
            agg_test = gmat.dot(sample.adj).dot(gmat.T)
            assert np.abs((agg_test - agg_adj).todense()).max() < 1e-6
            like = model.log_likelihood(sample)
            like = like / (model.num_vertices * (model.num_vertices - 1))
            assert like > -2.3
