""" Test the fitness model class on simple sample graph. """
import graph_ensembles.sparse as ge
import numpy as np
import pandas as pd
import pytest
import re

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

g = ge.MultiDiGraph(v, e, v_id=['name', 'country'],
                    src=['creditor', 'c_country'],
                    dst=['debtor', 'd_country'],
                    edge_label=['type', 'EUR'],
                    weight='value')

# Define graph marginals to check computation
out_strength = np.array([[0.0e+00, 4.0e+05, 1.0e+04, 0.0e+00],
                         [2.3e+07, 0.0e+00, 0.0e+00, 0.0e+00],
                         [0.0e+00, 0.0e+00, 3.0e+03, 7.0e+05],
                         [0.0e+00, 0.0e+00, 1.0e+06, 0.0e+00]])

in_strength = np.array([[2.300e+07, 0.000e+00, 1.003e+06, 7.000e+05],
                        [0.000e+00, 0.000e+00, 1.000e+04, 0.000e+00],
                        [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                        [0.000e+00, 4.000e+05, 0.000e+00, 0.000e+00]])

num_vertices = 4
num_labels = 4
num_edges = 5
num_edges_label = np.array([1, 1, 3, 1])

z = 8.98906315567293e-10

# Define p_ref for testing purposes (from z)
p_ref = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
p_ref[2, 0, 1] = 0.9988920931016365
p_ref[2, 0, 2] = 0.8998905002291506
p_ref[2, 1, 2] = 0.08247673514246777
p_ref[2, 3, 1] = 0.730080534269288
p_ref[2, 3, 2] = 0.026259053227431942
p_ref[0, 2, 1] = 0.9999978970495554
p_ref[3, 3, 1] = 0.9977348098283829
p_ref[1, 1, 0] = 0.993095114412297

z_self = 1.1263141e-10

# Define p_ref for testing purposes (from z_self)
p_self = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)


z_label = np.array(
    [2.02975770e-06, 6.71088639e-03, 1.82652908e-09, 2.19130984e-03])

# Define p_ref for testing purposes (from z_label)
p_ref_lbl = np.zeros((num_labels, num_vertices, num_vertices),
                     dtype=np.float64)
p_ref_lbl[2, 0, 1] = 0.9994544473260852
p_ref_lbl[2, 0, 2] = 0.9480930421837465
p_ref_lbl[2, 1, 2] = 0.15444301301041627
p_ref_lbl[2, 3, 1] = 0.846059367245662
p_ref_lbl[2, 3, 2] = 0.051949130017326955
p_ref_lbl[0, 2, 1] = 0.9999923706064989
p_ref_lbl[3, 3, 1] = 0.999992370604421
p_ref_lbl[1, 1, 0] = 0.9999923706050031

z_lbl_self = np.array(
    [2.02975770e-06, 6.71088639e-03, 2.95647709e-10, 2.19130984e-03])

# Define p_ref for testing purposes (from z_label)
p_self_lbl = np.zeros((num_labels, num_vertices, num_vertices),
                      dtype=np.float64)


class TestFitnessModelInit():
    def test_issubclass(self):
        """ Check that the model is a graph ensemble."""
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
        """ Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     num_edges=num_edges)
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert not model.per_label

        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     num_edges_label=num_edges_label)
        # assert np.all(model.prop_out == out_strength)
        # assert np.all(model.prop_in == in_strength)
        assert np.all(model.num_edges_label == num_edges_label)
        assert np.all(model.num_vertices == num_vertices)
        assert np.all(model.num_labels == num_labels)
        assert model.per_label

    def test_model_init_z(self):
        """ Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     param=z)
        assert np.all(model.param == z)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        np.testing.assert_allclose(model.expected_num_edges(),
                                   num_edges,
                                   rtol=1e-5)

        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     param=z_label)
        assert np.all(model.param == z_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        np.testing.assert_allclose(model.expected_num_edges_label(),
                                   num_edges_label,
                                   rtol=1e-5)

    def test_model_init_z_selfloops(self):
        """ Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     param=z_self, 
                                     selfloops=True)
        assert np.all(model.param == z_self)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        assert model.selfloops
        np.testing.assert_allclose(model.expected_num_edges(),
                                   num_edges,
                                   rtol=1e-5)

        model = ge.MultiFitnessModel(num_vertices=num_vertices,
                                     num_labels=num_labels,
                                     prop_out=out_strength,
                                     prop_in=in_strength,
                                     param=z_lbl_self,
                                     selfloops=True)
        assert np.all(model.param == z_lbl_self)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        assert model.selfloops
        np.testing.assert_allclose(model.expected_num_edges_label(),
                                   num_edges_label,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the model raises exceptions for wrong inputs."""
        msg = 'First argument passed must be a MultiDiGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel('df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.MultiFitnessModel(g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_nodes=num_vertices,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param=z_label)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=np.array([1, 2]),
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param=z_label)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=-3,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param=z_label)

    def test_wrong_num_labels(self):
        """ Check that wrong initialization of num_labels results in an error.
        """
        msg = 'Number of labels not set.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = re.escape('Node out properties must be a two dimensional array '
                        'with shape (num_vertices, num_labels).')
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=2,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of labels must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=np.array([1, 2]),
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of labels must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=-5,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of strengths results in an error.
        """
        msg = 'prop_out not set.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = 'prop_in not set.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 num_edges=num_edges)

        msg = re.escape('Node out properties must be a two dimensional array '
                        'with shape (num_vertices, num_labels).')
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=1,
                                 prop_in=in_strength,
                                 num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength[0:2, 0:1],
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = re.escape('Node in properties must be a two dimensional array '
                        'with shape (num_vertices, num_labels).')
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=3,
                                 num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength[0:2, 0:1],
                                 num_edges=num_edges)

        msg = "Node out properties must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=-out_strength,
                                 prop_in=in_strength,
                                 num_edges=num_edges)

        msg = "Node in properties must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=-in_strength,
                                 num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = 'num_edges must be an array of size one or num_labels, not 2.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=np.array([1, 2]))

        msg = 'Number of edges must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges='3')

        msg = 'Number of edges must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 num_edges=-23)

        msg = re.escape('Either num_edges(_label) or param must be set.')
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = 'The model requires one or num_labels parameter.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param=np.array([0, 1]))

        msg = 'Parameters must be numeric.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param='0')

        msg = 'Parameters must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.MultiFitnessModel(num_vertices=num_vertices,
                                 num_labels=num_labels,
                                 prop_out=out_strength,
                                 prop_in=in_strength,
                                 param=-1)


class TestFitnessModelFit():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(method="density")
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=False)
        model.fit(x0=1e2)
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_per_label(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_label, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init_per_label(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            z_label[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_bad_init_per_label(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            z_label[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """ Check that it raises an error with a negative initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=True)
        msg = "x0 must be positive."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=-1)

        msg = 'The model requires one or num_labels initial conditions.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0, 1]))

        msg = 'The model requires one or num_labels initial conditions.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0='hi')

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.MultiFitnessModel(g, per_label=True)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")


class TestFitnessModelFitSelfloops():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_with_bad_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=False, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(num_edges, model.expected_num_edges(),
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_self, model.param, atol=0, rtol=1e-6)

    def test_solver_newton_per_label(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(method="density")
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_init_per_label(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(x0=1e-14)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)

    def test_solver_with_bad_init_per_label(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.MultiFitnessModel(g, per_label=True, selfloops=True)
        model.fit(x0=1e2)
        np.testing.assert_allclose(
            num_edges_label, model.expected_num_edges_label(), 
            atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            z_lbl_self[2], model.param[2], atol=0, rtol=1e-6)


# class TestFitnessModelMeasures():

#     model = ge.FitnessModel(num_vertices=num_vertices,
#                             prop_out=out_strength,
#                             prop_in=in_strength,
#                             param=z)
    
#     def test_exp_n_edges(self):
#         """ Check expected edges is correct. """
#         ne = self.model.expected_num_edges()
#         np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

#     def test_exp_degree(self):
#         """ Check expected d is correct. """

#         d = self.model.expected_degree()
#         d_ref = (1 - (1 - p_ref)*(1 - p_ref.T))  # Only valid if no self loops
#         np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

#     def test_exp_out_degree(self):
#         """ Check expected d_out is correct. """
 
#         d_out = self.model.expected_out_degree()
#         np.testing.assert_allclose(d_out, p_ref.sum(axis=1), rtol=1e-5)
#         np.testing.assert_allclose(num_edges, np.sum(d_out))

#     def test_exp_in_degree(self):
#         """ Check expected d_out is correct. """

#         d_in = self.model.expected_in_degree()
#         np.testing.assert_allclose(d_in, p_ref.sum(axis=0), rtol=1e-5)
#         np.testing.assert_allclose(num_edges, np.sum(d_in))

#     def test_av_nn_prop_ones(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.ones(num_vertices)
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, np.array([1, 1, 0, 1]), 
#                                    atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#     def test_av_nn_prop_zeros(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.zeros(num_vertices)
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#     def test_av_nn_prop_scale(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.arange(num_vertices) + 1
#         p_u = (1 - (1 - p_ref)*(1 - p_ref.T))  # Only valid if no self loops
#         d = p_u.sum(axis=0)
#         d_out = p_ref.sum(axis=1)
#         d_in = p_ref.sum(axis=0)

#         exp = np.dot(p_ref, prop)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

#         exp = np.dot(p_ref.T, prop)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)
        
#         exp = np.dot(p_u, prop)
#         exp[d != 0] = exp[d != 0] / d[d != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

#     def test_av_nn_deg(self):
#         """ Test average nn degree."""

#         d_out = self.model.expected_out_degree()
#         d_in = self.model.expected_in_degree()

#         self.model.expected_av_nn_degree(ddir='out', ndir='out')
#         exp = np.dot(p_ref, d_out)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_nn_d_out, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='out', ndir='in')
#         exp = np.dot(p_ref.T, d_out)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         np.testing.assert_allclose(self.model.exp_av_in_nn_d_out, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='in', ndir='in')
#         exp = np.dot(p_ref.T, d_in)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='in', ndir='out')
#         exp = np.dot(p_ref, d_in)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_nn_d_in, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='out-in', ndir='out-in')
#         d = self.model.expected_degree()
#         exp = np.dot((1 - (1 - p_ref)*(1 - p_ref.T)), d)
#         exp[d != 0] = exp[d != 0] / d[d != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_in_nn_d_out_in, exp,
#                                    atol=1e-5, rtol=0)

#     def test_likelihood(self):
#         """ Test likelihood code. """
#         # Compute reference from p_ref
#         p_log = p_ref.copy()
#         p_log[p_log != 0] = np.log(p_log[p_log != 0])
#         np_log = np.log1p(-p_ref)
#         adj = np.array([[0, 1, 0, 1],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         ref = 0
#         for i in range(adj.shape[0]):
#             for j in range(adj.shape[1]):
#                 if adj[i, j] != 0:
#                     ref += p_log[i, j]
#                 else:
#                     ref += np_log[i, j]

#         # Construct model

#         np.testing.assert_allclose(ref, self.model.log_likelihood(g), 
#                                    atol=1e-6, rtol=1e-6)
#         np.testing.assert_allclose(ref, self.model.log_likelihood(
#             g.adjacency_matrix()), atol=1e-6, rtol=1e-6)
#         np.testing.assert_allclose(ref, self.model.log_likelihood(adj), 
#                                    atol=1e-6, rtol=1e-6)

#     def test_likelihood_inf_p_one(self):
#         """ Test likelihood code. """
#         # Construct adj with p[g] = 0
#         adj = np.array([[0, 1, 0, 1],
#                         [0, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         # Construct model
#         model = ge.FitnessModel(num_vertices=num_vertices,
#                                 prop_out=out_strength,
#                                 prop_in=in_strength,
#                                 param=np.array([np.infty]))

#         res = model.log_likelihood(adj)
#         assert np.isinf(res) and (res < 0)

#     def test_likelihood_inf_p_zero(self):
#         """ Test likelihood code. """
#         # Construct adj with p[g] = 0
#         adj = np.array([[0, 1, 1, 1],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         # Construct model

#         res = self.model.log_likelihood(adj)
#         assert np.isinf(res) and (res < 0)

#     def test_likelihood_error(self):
#         """ Test likelihood code. """
#         adj = np.array([[0, 1, 0, 0],
#                         [1, 0, 1, 0],
#                         [0, 1, 0, 0]])

#         # Construct model

#         msg = re.escape('Passed graph adjacency matrix does not have the '
#                         'correct shape: (3, 4) instead of (4, 4)')
#         with pytest.raises(ValueError, match=msg):
#             self.model.log_likelihood(adj)

#         msg = 'g input not a graph or adjacency matrix.'
#         with pytest.raises(ValueError, match=msg):
#             self.model.log_likelihood('dfsg')


# class TestFitnessModelMeasuresSelfloops():

#     model = ge.FitnessModel(num_vertices=num_vertices,
#                             prop_out=out_strength,
#                             prop_in=in_strength,
#                             param=z_self, selfloops=True)
    
#     def test_exp_n_edges(self):
#         """ Check expected edges is correct. """
#         ne = self.model.expected_num_edges()
#         np.testing.assert_allclose(ne, num_edges, rtol=1e-5)

#     def test_exp_degree(self):
#         """ Check expected d is correct. """

#         d = self.model.expected_degree()
#         d_ref = (1 - (1 - p_self)*(1 - p_self.T)) 
#         d_ref.ravel()[::num_vertices+1] = p_self.ravel()[::num_vertices+1]
#         np.testing.assert_allclose(d, d_ref.sum(axis=0), rtol=1e-5)

#     def test_exp_out_degree(self):
#         """ Check expected d_out is correct. """
 
#         d_out = self.model.expected_out_degree()
#         np.testing.assert_allclose(d_out, p_self.sum(axis=1), rtol=1e-5)
#         np.testing.assert_allclose(num_edges, np.sum(d_out))

#     def test_exp_in_degree(self):
#         """ Check expected d_out is correct. """

#         d_in = self.model.expected_in_degree()
#         np.testing.assert_allclose(d_in, p_self.sum(axis=0), rtol=1e-5)
#         np.testing.assert_allclose(num_edges, np.sum(d_in))

#     def test_av_nn_prop_ones(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.ones(num_vertices)
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, np.array([1, 1, 0, 1]), 
#                                    atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#     def test_av_nn_prop_zeros(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.zeros(num_vertices)
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

#     def test_av_nn_prop_scale(self):
#         """ Test correct value of av_nn_prop using simple local prop. """

#         prop = np.arange(num_vertices) + 1
#         p_u = (1 - (1 - p_self)*(1 - p_self.T)) 
#         p_u.ravel()[::num_vertices+1] = p_self.ravel()[::num_vertices+1]
#         d = p_u.sum(axis=0)
#         d_out = p_self.sum(axis=1)
#         d_in = p_self.sum(axis=0)

#         exp = np.dot(p_self, prop)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='out')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

#         exp = np.dot(p_self.T, prop)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='in')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)
        
#         exp = np.dot(p_u, prop)
#         exp[d != 0] = exp[d != 0] / d[d != 0]
#         res = self.model.expected_av_nn_property(prop, ndir='out-in')
#         np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

#     def test_av_nn_deg(self):
#         """ Test average nn degree."""

#         d_out = self.model.expected_out_degree()
#         d_in = self.model.expected_in_degree()

#         self.model.expected_av_nn_degree(ddir='out', ndir='out')
#         exp = np.dot(p_self, d_out)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_nn_d_out, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='out', ndir='in')
#         exp = np.dot(p_self.T, d_out)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         np.testing.assert_allclose(self.model.exp_av_in_nn_d_out, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='in', ndir='in')
#         exp = np.dot(p_self.T, d_in)
#         exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
#         np.testing.assert_allclose(self.model.exp_av_in_nn_d_in, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='in', ndir='out')
#         exp = np.dot(p_self, d_in)
#         exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_nn_d_in, exp,
#                                    atol=1e-5, rtol=0)

#         self.model.expected_av_nn_degree(ddir='out-in', ndir='out-in')
#         d = self.model.expected_degree()
#         p_u = (1 - (1 - p_self)*(1 - p_self.T)) 
#         p_u.ravel()[::num_vertices+1] = p_self.ravel()[::num_vertices+1]
#         exp = np.dot(p_u, d)
#         exp[d != 0] = exp[d != 0] / d[d != 0]
#         np.testing.assert_allclose(self.model.exp_av_out_in_nn_d_out_in, exp,
#                                    atol=1e-5, rtol=0)

#     def test_likelihood(self):
#         """ Test likelihood code. """
#         # Compute reference from p_self
#         p_log = p_self.copy()
#         p_log[p_log != 0] = np.log(p_log[p_log != 0])
#         np_log = np.log1p(-p_self)
#         adj = np.array([[0, 1, 0, 1],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         ref = 0
#         for i in range(adj.shape[0]):
#             for j in range(adj.shape[1]):
#                 if adj[i, j] != 0:
#                     ref += p_log[i, j]
#                 else:
#                     ref += np_log[i, j]

#         np.testing.assert_allclose(ref, self.model.log_likelihood(g), 
#                                    atol=1e-6, rtol=1e-6)
#         np.testing.assert_allclose(ref, self.model.log_likelihood(
#             g.adjacency_matrix()), atol=1e-6, rtol=1e-6)
#         np.testing.assert_allclose(ref, self.model.log_likelihood(adj), 
#                                    atol=1e-6, rtol=1e-6)

#     def test_likelihood_inf_p_one(self):
#         """ Test likelihood code. """
#         # Construct adj with p[g] = 0
#         adj = np.array([[0, 1, 0, 1],
#                         [0, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         # Construct model
#         model = ge.FitnessModel(num_vertices=num_vertices,
#                                 prop_out=out_strength,
#                                 prop_in=in_strength,
#                                 param=np.array([np.infty]))

#         res = model.log_likelihood(adj)
#         assert np.isinf(res) and (res < 0)

#     def test_likelihood_inf_p_zero(self):
#         """ Test likelihood code. """
#         # Construct adj with p[g] = 0
#         adj = np.array([[0, 1, 1, 1],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]])

#         res = self.model.log_likelihood(adj)
#         assert np.isinf(res) and (res < 0)

#     def test_likelihood_error(self):
#         """ Test likelihood code. """
#         adj = np.array([[0, 1, 0, 0],
#                         [1, 0, 1, 0],
#                         [0, 1, 0, 0]])

#         msg = re.escape('Passed graph adjacency matrix does not have the '
#                         'correct shape: (3, 4) instead of (4, 4)')
#         with pytest.raises(ValueError, match=msg):
#             self.model.log_likelihood(adj)

#         msg = 'g input not a graph or adjacency matrix.'
#         with pytest.raises(ValueError, match=msg):
#             self.model.log_likelihood('dfsg')


# class TestFitnessModelSample():
#     def test_sampling(self):
#         """ Check that properties of the sample correspond to ensemble.
#         """
#         model = ge.FitnessModel(num_vertices=num_vertices,
#                                 prop_out=out_strength,
#                                 prop_in=in_strength,
#                                 param=z)

#         samples = 100
#         for i in range(samples):
#             sample = model.sample()
#             like = model.log_likelihood(sample) 
#             like = like / (model.num_vertices * (model.num_vertices - 1))
#             assert like > -2.3

#     def test_sampling_selfloops(self):
#         """ Check that properties of the sample correspond to ensemble.
#         """
#         model = ge.FitnessModel(num_vertices=num_vertices,
#                                 prop_out=out_strength,
#                                 prop_in=in_strength,
#                                 param=z_self,
#                                 selfloops=True)

#         samples = 100
#         for i in range(samples):
#             sample = model.sample()
#             like = model.log_likelihood(sample) 
#             like = like / (model.num_vertices * (model.num_vertices - 1))
#             assert like > -2.3
