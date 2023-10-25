""" Test the fitness model class on simple sample graph. """
import graph_ensembles.spark as ge
import numpy as np
import pandas as pd
import pytest
import re
from pyspark import SparkContext


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

e_l = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                   ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                   ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                   ['BNP', 'IT', 'ABN', 'NL', 3e3, 'interbank', False],
                   ['ABN', 'NL', 'BNP', 'FR', 1e4, 'interbank', False],
                   ['ABN', 'NL', 'ING', 'NL', 4e5, 'external', True]],
                   columns=['creditor', 'c_country',
                            'debtor', 'd_country',
                            'value', 'type', 'EUR'])

g = ge.DiGraph(v, e, v_id=['name', 'country'],
               src=['creditor', 'c_country'],
               dst=['debtor', 'd_country'],
               weight='value')

g_l = ge.MultiDiGraph(v, e_l, v_id=['name', 'country'],
                      src=['creditor', 'c_country'],
                      dst=['debtor', 'd_country'],
                      edge_label=['type', 'EUR'],
                      weight='value')

# Define graph marginals to check computation
out_strength = np.array([1e4 + 4e5, 2.3e7, 7e5 + 3e3, 1e6],
                        dtype=np.float64)

in_strength = np.array([1e6 + 3e3 + 2.3e7 + 7e5,  1e4, 0, 4e5],
                       dtype=np.float64)

num_vertices = 4
num_edges = 5
z = 2.91360376e-12

p_ref = np.array([[0.00000000, 0.01180476, 0.00000000, 0.32333265],
                  [0.99939629, 0.00000000, 0.00000000, 0.96403545],
                  [0.98061950, 0.02007152, 0.00000000, 0.45033967],
                  [0.98629663, 0.02831116, 0.00000000, 0.00000000]])

# Initialize spark
sc = SparkContext()


class TestFitnessModelInit():

    def test_issubclass(self):
        """ Check that the model is a graph ensemble."""
        model = ge.ScaleInvariantModel(sc, g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.ScaleInvariantModel(sc, g)
        assert np.all(model.fit_out == out_strength), g.out_strength()
        assert np.all(model.fit_in == in_strength), g.in_strength()
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

        model = ge.ScaleInvariantModel(sc, g)
        assert np.all(model.fit_out == out_strength)
        assert np.all(model.fit_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
        """ Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                num_edges=num_edges)
        assert np.all(model.fit_out == out_strength)
        assert np.all(model.fit_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_z(self):
        """ Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)
        assert np.all(model.fit_out == out_strength)
        assert np.all(model.fit_in == in_strength)
        assert np.all(model.param == z)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.expected_num_edges(),
                                   num_edges,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the model raises exceptions for wrong inputs."""
        msg = 'First argument must be a SparkContext.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(g, 'df', 234, out_strength)

        msg = 'A SparkContext must be passed as the first argument.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(fit_out=out_strength, fit_in=in_strength)

        msg = 'Second argument passed must be a DiGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc, 'df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.ScaleInvariantModel(sc, g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_nodes=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc, 
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=num_edges)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=np.array([1, 2]),
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=num_edges)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=-3,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of strengths results in an error.
        """
        msg = 'fit_out not set.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_in=in_strength,
                            num_edges=num_edges)

        msg = 'fit_in not set.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            num_edges=num_edges)

        msg = ("Out fitness must be a numpy array of length " +
               str(num_vertices))
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=1,
                            fit_in=in_strength,
                            num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength[0:2],
                            fit_in=in_strength,
                            num_edges=num_edges)

        msg = ("In fitness must be a numpy array of length " +
               str(num_vertices))
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=2,
                            num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength[0:2],
                            num_edges=num_edges)

        msg = "Out fitness must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=-out_strength,
                            fit_in=in_strength,
                            num_edges=num_edges)

        msg = "In fitness must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=-in_strength,
                            num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = 'Number of edges must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=np.array([1, 2]))

        msg = 'Number of edges must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges='3')

        msg = 'Number of edges must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            num_edges=-324)

        msg = 'Either num_edges or param must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = 'The FitnessModel requires one parameter.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            param=np.array([0, 1]))

        msg = 'Parameters must be numeric.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            param='0')

        msg = 'Parameters must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.ScaleInvariantModel(sc,
                            num_vertices=num_vertices,
                            fit_out=out_strength,
                            fit_in=in_strength,
                            param=-1)


class TestFitnessModelFit():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.ScaleInvariantModel(sc, g)
        model.fit(method="density", atol=1e-6)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.ScaleInvariantModel(sc, g)
        model.fit(x0=1e-14)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.ScaleInvariantModel(sc, g)
        model.fit(x0=1e2)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """ Check that it raises an error with a negative initial condition.
        """
        model = ge.ScaleInvariantModel(sc, g)
        msg = "x0 must be positive."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=-1)

        msg = 'The FitnessModel requires one parameter.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0, 1]))

        msg = 'x0 must be numeric.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0='hi')

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.ScaleInvariantModel(sc, g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")


class TestFitnessModelMeasures():
    def test_exp_n_edges(self):
        """ Check expected edges is correct. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)
        ne = model.expected_num_edges()
        np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

    def test_exp_degree(self):
        """ Check expected d is correct. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        d = model.expected_degree()
        d_ref = (1 - (1 - p_ref)*(1 - p_ref.T))  # Only valid if no self loops
        np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

    def test_exp_out_degree(self):
        """ Check expected d_out is correct. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)
 
        d_out = model.expected_out_degree()
        np.testing.assert_allclose(d_out, p_ref.sum(axis=1), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_out))

    def test_exp_in_degree(self):
        """ Check expected d_out is correct. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        d_in = model.expected_in_degree()
        np.testing.assert_allclose(d_in, p_ref.sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_in))

    def test_av_nn_prop_ones(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        prop = np.ones(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, np.array([1, 1, 0, 1]), 
                                   atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_zeros(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        prop = np.zeros(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_scale(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        prop = np.arange(num_vertices) + 1
        p_u = (1 - (1 - p_ref)*(1 - p_ref.T))  # Only valid if no self loops
        d = p_u.sum(axis=0)
        d_out = p_ref.sum(axis=1)
        d_in = p_ref.sum(axis=0)

        exp = np.dot(p_ref, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_ref.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)
        
        exp = np.dot(p_u, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg(self):
        """ Test average nn degree."""
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        d_out = model.expected_out_degree()
        d_in = model.expected_in_degree()

        model.expected_av_nn_degree(ddir='out', ndir='out')
        exp = np.dot(p_ref, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out', ndir='in')
        exp = np.dot(p_ref.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='in')
        exp = np.dot(p_ref.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='out')
        exp = np.dot(p_ref, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out-in', ndir='out-in')
        d = model.expected_degree()
        exp = np.dot((1 - (1 - p_ref)*(1 - p_ref.T)), d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(model.exp_av_out_in_nn_d_out_in, exp,
                                   atol=1e-5, rtol=0)

    def test_likelihood(self):
        """ Test likelihood code. """
        # Compute reference from p_ref
        p_log = p_ref.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.log(1 - p_ref)
        adj = np.array([[0, 1, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]])

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ref += p_log[i, j]
                else:
                    ref += np_log[i, j]

        # Construct model
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        assert np.abs(ref - model.log_likelihood(g)) < 1e-6
        assert np.abs(ref - model.log_likelihood(g.adjacency_matrix())) < 1e-6
        assert np.abs(ref - model.log_likelihood(adj)) < 1e-6

    def test_likelihood_error(self):
        """ Test likelihood code. """
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 0]])

        # Construct model
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        msg = re.escape('Passed graph adjacency matrix does not have the '
                        'correct shape: (3, 4) instead of (4, 4)')
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood(adj)

        msg = 'g input not a graph or adjacency matrix.'
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood('dfsg')


class TestFitnessModelSample():
    def test_sampling(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.ScaleInvariantModel(sc,
                                num_vertices=num_vertices,
                                fit_out=out_strength,
                                fit_in=in_strength,
                                param=z)

        samples = 100
        s_n_e = np.empty(samples)
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges()

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges, atol=1e-1, rtol=0)
