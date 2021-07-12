
""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
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

g = ge.Graph(v, e, v_id=['name', 'country'],
             src=['creditor', 'c_country'],
             dst=['debtor', 'd_country'],
             edge_label=['type', 'EUR'],
             weight='value')

# Define graph marginals to check computation
out_strength = np.rec.array([(0, 0, 1e6),
                             (0, 1, 1e4),
                             (0, 3, 3e3),
                             (1, 2, 2.3e7),
                             (2, 3, 7e5),
                             (3, 1, 4e5)],
                            dtype=[('label', np.uint8),
                                   ('id', np.uint8),
                                   ('value', np.float64)])

in_strength = np.rec.array([(0, 1, 1e6 + 3e3),
                            (0, 2, 1e4),
                            (1, 1, 2.3e7),
                            (2, 1, 7e5),
                            (3, 0, 4e5)],
                           dtype=[('label', np.uint8),
                                  ('id', np.uint8),
                                  ('value', np.float64)])

num_vertices = 4
num_edges = 5
num_edges_label = np.array([3, 1, 1, 1])
num_labels = 4
z = np.array([8.989062e-10])
z_label = np.array([1.826524e-09, 2.477713e-10, 2.674918e-07, 8.191937e-07])
z_inv = np.array([3.19467e-10])
z_inv_lbl = np.array([7.749510e-10, 2.268431e-14, 2.448980e-11, 7.500000e-11])

# Define p_ref for testing purposes (from z)
p_ref = np.zeros((num_labels, num_vertices, num_vertices), dtype=np.float64)
p_ref[0, 0, 1] = 0.9988920931016365
p_ref[0, 0, 2] = 0.8998905002291506
p_ref[0, 1, 2] = 0.08247673514246777
p_ref[0, 3, 1] = 0.730080534269288
p_ref[0, 3, 2] = 0.026259053227431942
p_ref[1, 2, 1] = 0.9999978970495554
p_ref[2, 3, 1] = 0.9977348098283829
p_ref[3, 1, 0] = 0.993095114412297

# Define p_ref for testing purposes (from z_label)
p_ref_lbl = np.zeros((num_labels, num_vertices, num_vertices),
                     dtype=np.float64)
p_ref_lbl[0, 0, 1] = 0.9994544473260852
p_ref_lbl[0, 0, 2] = 0.9480930421837465
p_ref_lbl[0, 1, 2] = 0.15444301301041627
p_ref_lbl[0, 3, 1] = 0.846059367245662
p_ref_lbl[0, 3, 2] = 0.051949130017326955
p_ref_lbl[1, 2, 1] = 0.9999923706064989
p_ref_lbl[2, 3, 1] = 0.999992370604421
p_ref_lbl[3, 1, 0] = 0.9999923706050031


class TestStripeFitnessModelInit():
    def test_issubclass(self):
        """ Check that the stripe model is a graph ensemble."""
        model = ge.StripeFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init_g(self):
        """ Check that the stripe model can be correctly initialized from
        a graph object.
        """
        model = ge.StripeFitnessModel(g, per_label=False)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label

        model = ge.StripeFitnessModel(g, per_label=True)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label

    def test_model_init(self):
        """ Check that the stripe model can be correctly initialized from
        parameters directly.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      num_edges=num_edges)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      num_edges=num_edges_label)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label

    def test_model_init_param(self):
        """ Check that the stripe model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.param == z)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-5)

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.param == z_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        np.testing.assert_allclose(model.num_edges,
                                   num_edges_label,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the stripe model raises exceptions for wrong inputs."""
        msg = 'First argument passed must be a WeightedLabelGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel('df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.StripeFitnessModel(g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_nodes=num_vertices,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices smaller than max id value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=2,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=np.array([1, 2]),
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=-3,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_num_labels(self):
        """ Check that wrong initialization of num_labels results in an error.
        """
        msg = 'Number of labels not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels smaller than max label value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=2,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=np.array([1, 2]),
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=-5,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = 'out_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'in_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  num_edges=num_edges)

        msg = re.escape("Out strength must be a rec array with columns: "
                        "('label', 'id', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=1,
                                  in_strength=in_strength,
                                  num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength.value,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = re.escape("In strength must be a rec array with columns: "
                        "('label', 'id', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=2,
                                  num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength.value,
                                  num_edges=num_edges)

        msg = "Sums of strengths per label do not match."
        tmp = out_strength.copy()
        tmp.value[0] = tmp.value[0] + 1
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = "Storing zeros in the strengths leads to inefficient code."
        tmp = out_strength.copy()
        tmp.resize(len(tmp) + 1)
        tmp[-1] = ((1, 2, 0))
        with pytest.warns(UserWarning, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_out_strength(self):
        """ Test that an error is raised if out_strength contains negative
        values in either id, label or value.
        """
        tmp = out_strength.copy().astype([('label', np.int8),
                                          ('id', np.int8),
                                          ('value', np.float64)])

        tmp.label[1] = -1
        msg = "Out strength labels must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.label[1] = out_strength.label[1]
        tmp.id[2] = -tmp.id[2]
        msg = "Out strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "Out strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_in_strength(self):
        """ Test that an error is raised if in_strength contains negative
        values in either id, label or value.
        """
        tmp = in_strength.copy().astype([('label', np.int8),
                                         ('id', np.int8),
                                         ('value', np.float64)])

        tmp.label[2] = -tmp.label[2]
        msg = "In strength labels must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.label[2] = -tmp.label[2]
        tmp.id[2] = -tmp.id[2]
        msg = "In strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "In strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = ('Number of edges must be a number or a numpy array of length'
               ' equal to the number of labels.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges='3')
        with pytest.raises(Exception, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=np.array([1, 2]))

        msg = 'Number of edges must contain only positive values.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=-324)
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=-num_edges_label)

        msg = 'Either num_edges or param must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = ('StripeFitnessModel requires an array of parameters with number'
               ' of elements equal to the number of labels or to one.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  param=np.array([0, 1]))

        msg = 'Parameters must be numeric.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  param='0')

        msg = 'Parameters must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  param=-1)
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  param=-z_label)


class TestStripeFitnessModelFit():
    def test_solver_newton_single_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.StripeFitnessModel(g, per_label=False)
        model.fit(method="newton")
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param[0], atol=0, rtol=1e-6)

    def test_solver_newton_multi_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.StripeFitnessModel(g)
        model.fit(method="newton")
        model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, model.exp_num_edges_label,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_label, model.param[0], atol=0, rtol=1e-5)

    def test_solver_invariant_single_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly for the invariant case. """
        model = ge.StripeFitnessModel(g, per_label=False, scale_invariant=True)
        model.fit(method="newton")
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_inv, model.param[0], atol=0, rtol=1e-6)

    def test_solver_invariant_multi_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly for the invariant case. """
        model = ge.StripeFitnessModel(g, scale_invariant=True)
        model.fit(method="newton")
        model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, model.exp_num_edges_label,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_inv_lbl, model.param[0],
                                   atol=0, rtol=1e-6)

    def test_solver_fixed_point_multi_z(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.StripeFitnessModel(g)
        model.fit(method="fixed-point", max_iter=50000, xtol=1e-4)
        model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, model.exp_num_edges_label,
                                   atol=1e-3, rtol=0)

    def test_solver_min_degree_single_z(self):
        """ Check that the min_degree solver converges.
        """
        out_strength = np.rec.array([(0, 0, 2e9),
                                     (0, 1, 5e8)],
                                    dtype=[('label', np.uint8),
                                           ('id', np.uint8),
                                           ('value', np.float64)])

        in_strength = np.rec.array([(0, 2, 1e9),
                                    (0, 3, 1e9),
                                    (0, 4, 5e8)],
                                   dtype=[('label', np.uint8),
                                          ('id', np.uint8),
                                          ('value', np.float64)])

        num_vertices = 5
        num_edges = 3
        num_labels = 1

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      num_edges=num_edges,
                                      min_degree=True)

        model.fit(x0=np.array([[1e-6, 0.1]], dtype=np.float64).T,
                  tol=1e-6,
                  max_iter=500,
                  verbose=True)
        model.expected_num_edges()
        assert np.abs(num_edges - model.exp_num_edges) < 1e-5
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree
        assert np.all(d_out[d_out != 0] >= 1 - 1e-5)
        assert np.all(d_in[d_in != 0] >= 1 - 1e-5)

    def test_solver_min_degree_multi_z(self):
        """ Check that the min_degree solver converges.
        """
        e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e4, 'interbank', False],
                         ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                         ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                         ['BNP', 'IT', 'ABN', 'NL', 3e3, 'interbank', False],
                         ['ABN', 'NL', 'BNP', 'FR', 1e6, 'interbank', False],
                         ['ING', 'NL', 'BNP', 'IT', 3e6, 'interbank', False],
                         ['ABN', 'NL', 'ING', 'NL', 4e5, 'external', True]],
                         columns=['creditor', 'c_country',
                                  'debtor', 'd_country',
                                  'value', 'type', 'EUR'])

        g = ge.Graph(v, e, v_id=['name', 'country'],
                     src=['creditor', 'c_country'],
                     dst=['debtor', 'd_country'],
                     edge_label=['type', 'EUR'],
                     weight='value')

        model = ge.StripeFitnessModel(g, min_degree=True)
        model.fit(x0=np.array([[1e-4, 1e-4, 1e-4, 1e-4],
                               [1, 1, 1, 1]]), 
                  tol=1e-6,
                  max_iter=500)
        model.expected_num_edges_label()
        np.testing.assert_allclose(np.array([4., 1., 1., 1.]),
                                   model.exp_num_edges_label,
                                   atol=1e-5, rtol=0)
        model.expected_degrees_by_label()
        exp_d_out = model.exp_out_degree_label.value
        exp_d_in = model.exp_in_degree_label.value
        assert np.all(exp_d_out >= 1 - 1e-5)
        assert np.all(exp_d_in >= 1 - 1e-5)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.StripeFitnessModel(g, per_label=False)
        model.fit(x0=1e-14)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param[0], atol=0, rtol=1e-6)

        model = ge.StripeFitnessModel(g, per_label=True)
        model.fit(x0=1e-14*np.ones(num_labels))
        model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, model.exp_num_edges_label,
                                   atol=1e-5, rtol=0)

    def test_solver_with_wrong_init(self):
        """ Check that it raises an error with a negative initial condition.
        """
        msg = 'Parameters must be positive.'
        with pytest.raises(ValueError, match=msg):
            model = ge.StripeFitnessModel(g, per_label=False)
            model.fit(x0=-1)
        with pytest.raises(ValueError, match=msg):
            model = ge.StripeFitnessModel(g, per_label=True)
            model.fit(x0=-np.ones(num_labels))

        msg = 'Parameters must be numeric.'
        with pytest.raises(ValueError, match=msg):
            model = ge.StripeFitnessModel(g, per_label=False)
            model.fit(x0='0')

        msg = ('StripeFitnessModel requires an array of parameters with number'
               ' of elements equal to the number of labels or to one.')
        with pytest.raises(AssertionError, match=msg):
            model = ge.StripeFitnessModel(g, per_label=True)
            model.fit(x0=np.array([0, 1]))

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.StripeFitnessModel(g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")

    def test_method_incompatibility(self):
        """ Check that an error is raised when trying to use the wrong method.
        """
        model = ge.StripeFitnessModel(g, scale_invariant=True)
        msg = ('Fixed point solver not supported for scale '
               'invariant functional.')
        with pytest.raises(Exception, match=msg):
            model.fit(method="fixed-point", max_iter=100, xtol=1e-5)

        # model = ge.StripeFitnessModel(g, min_degree=True)
        # msg = ('Method not recognised for solver with min degree '
        #        'constraint, using default SLSQP.')
        # with pytest.warns(UserWarning, match=msg):
        #     model.fit(method="newton")

        msg = 'Cannot constrain min degree in scale invariant model.'
        with pytest.raises(Exception, match=msg):
            model = ge.StripeFitnessModel(
              g, scale_invariant=True, min_degree=True)


class TestFitnessModelMeasures():
    def test_exp_n_edges_single_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        model.expected_num_edges()
        np.testing.assert_allclose(model.exp_num_edges,
                                   num_edges,
                                   rtol=1e-5)

    def test_exp_n_edges_multi_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        model.expected_num_edges()
        np.testing.assert_allclose(model.exp_num_edges,
                                   5.153923,
                                   rtol=1e-5)

    def test_exp_n_edges_label_single_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        model.expected_num_edges_label()
        np.testing.assert_allclose(model.exp_num_edges_label,
                                   np.array([2.737599, 0.999998,
                                             0.997735, 0.993095]),
                                   rtol=1e-5)

    def test_exp_n_edges_label_multi_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        model.expected_num_edges_label()
        np.testing.assert_allclose(model.exp_num_edges_label,
                                   num_edges_label,
                                   rtol=1e-5)

    def test_exp_out_degree_single_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        model.expected_degrees()
        d_out = model.exp_out_degree
        np.testing.assert_allclose(
            d_out, np.array([1.898792, 1.07558, 0.999998, 1.02565]),
            rtol=1e-5)

    def test_exp_out_degree_multi_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        model.expected_degrees()
        d_out = model.exp_out_degree
        np.testing.assert_allclose(
            d_out, np.array([1.947547, 1.154435, 0.999992, 1.051948]),
            rtol=1e-5)

    def test_exp_in_degree_single_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        model.expected_degrees()
        d_in = model.exp_in_degree
        np.testing.assert_allclose(
            d_in, np.array([0.993095, 2.998279, 1.008626, 0.0]), rtol=1e-5)

    def test_exp_in_degree_multi_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        model.expected_degrees()
        d_in = model.exp_in_degree
        np.testing.assert_allclose(
            d_in, np.array([0.999992, 2.999446, 1.154485, 0.0]), rtol=1e-5)

    def test_exp_out_degree_by_label_single_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        d_ref = np.array([(0, 0, 1.898783),
                          (0, 1, 0.082477),
                          (0, 3, 0.756340),
                          (1, 2, 0.999998),
                          (2, 3, 0.997735),
                          (3, 1, 0.993095)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        model.expected_degrees_by_label()
        d_out = model.exp_out_degree_label
        np.testing.assert_allclose(d_out.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_out.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_out.value, d_ref.value, rtol=1e-5)

    def test_exp_out_degree_by_label_multi_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        d_ref = np.array([(0, 0, 1.947547),
                          (0, 1, 0.154443),
                          (0, 3, 0.898008),
                          (1, 2, 0.999992),
                          (2, 3, 0.999992),
                          (3, 1, 0.999992)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        model.expected_degrees_by_label()
        d_out = model.exp_out_degree_label
        np.testing.assert_allclose(d_out.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_out.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_out.value, d_ref.value, rtol=1e-5)

    def test_exp_in_degree_by_label_single_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)
        d_ref = np.array([(0, 1, 1.728973),
                          (0, 2, 1.008626),
                          (1, 1, 0.999998),
                          (2, 1, 0.997735),
                          (3, 0, 0.993095)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        model.expected_degrees_by_label()
        d_in = model.exp_in_degree_label
        np.testing.assert_allclose(d_in.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_in.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_in.value, d_ref.value, rtol=1e-5)

    def test_exp_in_degree_by_label_multi_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)
        d_ref = np.array([(0, 1, 1.845514),
                          (0, 2, 1.154485),
                          (1, 1, 0.999992),
                          (2, 1, 0.999992),
                          (3, 0, 0.999992)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        model.expected_degrees_by_label()
        d_in = model.exp_in_degree_label
        np.testing.assert_allclose(d_in.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_in.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_in.value, d_ref.value, rtol=1e-5)

    def test_av_nn_prop_ones_single_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        prop = np.ones(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, np.array([1, 1, 1, 0]), 
                                   atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_ones_multi_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        prop = np.ones(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, np.array([1, 1, 1, 0]), 
                                   atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_zeros_single_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        prop = np.zeros(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_zeros_multi_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        prop = np.zeros(num_vertices)
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, prop, atol=1e-6, rtol=0)

    def test_av_nn_prop_scale_single_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        prop = np.arange(num_vertices) + 1
        p_c = 1 - np.prod(1 - p_ref, axis=0)
        p_u = (1 - (1 - p_c)*(1 - p_c.T))  # Only valid if no self loops
        d = p_u.sum(axis=0)
        d_out = p_c.sum(axis=1)
        d_in = p_c.sum(axis=0)

        exp = np.dot(p_c, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_c.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)
        
        exp = np.dot(p_u, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_prop_scale_multi_z(self):
        """ Test correct value of av_nn_prop using simple local prop. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        prop = np.arange(num_vertices) + 1
        p_c = 1 - np.prod(1 - p_ref_lbl, axis=0)
        p_u = (1 - (1 - p_c)*(1 - p_c.T))  # Only valid if no self loops
        d = p_u.sum(axis=0)
        d_out = p_c.sum(axis=1)
        d_in = p_c.sum(axis=0)

        exp = np.dot(p_c, prop)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        res = model.expected_av_nn_property(prop, ndir='out')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

        exp = np.dot(p_c.T, prop)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        res = model.expected_av_nn_property(prop, ndir='in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)
        
        exp = np.dot(p_u, prop)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        res = model.expected_av_nn_property(prop, ndir='out-in')
        np.testing.assert_allclose(res, exp, atol=1e-3, rtol=0)

    def test_av_nn_deg_single_z(self):
        """ Test average nn degree."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        p_c = 1 - np.prod(1 - p_ref, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree

        model.expected_av_nn_degree(ddir='out', ndir='out')
        exp = np.dot(p_c, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out', ndir='in')
        exp = np.dot(p_c.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='in')
        exp = np.dot(p_c.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='out')
        exp = np.dot(p_c, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out-in', ndir='out-in')
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)), d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(model.exp_av_out_in_nn_d_out_in, exp,
                                   atol=1e-5, rtol=0)

    def test_av_nn_deg_multi_z(self):
        """ Test average nn degree."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        p_c = 1 - np.prod(1 - p_ref_lbl, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree

        model.expected_av_nn_degree(ddir='out', ndir='out')
        exp = np.dot(p_c, d_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out', ndir='in')
        exp = np.dot(p_c.T, d_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_out, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='in')
        exp = np.dot(p_c.T, d_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='in', ndir='out')
        exp = np.dot(p_c, d_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_d_in, exp,
                                   atol=1e-5, rtol=0)

        model.expected_av_nn_degree(ddir='out-in', ndir='out-in')
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)), d)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(model.exp_av_out_in_nn_d_out_in, exp,
                                   atol=1e-5, rtol=0)

    def test_av_nn_strength_single_z(self):
        """ Test average nn strength."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        p_c = 1 - np.prod(1 - p_ref, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree
        s_out = ge.lib.to_sparse(out_strength, (num_vertices, num_labels),
                                 i_col='id', j_col='label', data_col='value',
                                 kind='csr').sum(axis=1).A1
        s_in = ge.lib.to_sparse(in_strength, (num_vertices, num_labels),
                                i_col='id', j_col='label', data_col='value',
                                kind='csr').sum(axis=1).A1

        model.expected_av_nn_strength(sdir='out', ndir='out')
        exp = np.dot(p_c, s_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_s_out, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='out')
        exp = np.dot(p_c, s_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_s_in, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='in')
        exp = np.dot(p_c.T, s_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_s_in, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out', ndir='in')
        exp = np.dot(p_c.T, s_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_s_out, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out-in', ndir='out-in')
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)),
                     s_out + s_in)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(model.exp_av_out_in_nn_s_out_in, exp,
                                   rtol=1e-6)

    def test_av_nn_strength_multi_z(self):
        """ Test average nn strength."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        p_c = 1 - np.prod(1 - p_ref_lbl, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree
        s_out = ge.lib.to_sparse(out_strength, (num_vertices, num_labels),
                                 i_col='id', j_col='label', data_col='value',
                                 kind='csr').sum(axis=1).A1
        s_in = ge.lib.to_sparse(in_strength, (num_vertices, num_labels),
                                i_col='id', j_col='label', data_col='value',
                                kind='csr').sum(axis=1).A1
        print(s_in)
        model.expected_av_nn_strength(sdir='out', ndir='out')
        exp = np.dot(p_c, s_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_s_out, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='out')
        exp = np.dot(p_c, s_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0]
        np.testing.assert_allclose(model.exp_av_out_nn_s_in, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='in')
        exp = np.dot(p_c.T, s_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_s_in, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out', ndir='in')
        exp = np.dot(p_c.T, s_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0]
        np.testing.assert_allclose(model.exp_av_in_nn_s_out, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out-in', ndir='out-in')
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)),
                     s_out + s_in)
        exp[d != 0] = exp[d != 0] / d[d != 0]
        np.testing.assert_allclose(model.exp_av_out_in_nn_s_out_in, exp,
                                   rtol=1e-6)

    def test_av_nn_strength_label_single_z(self):
        """ Test average nn strength."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        p_c = 1 - np.prod(1 - p_ref, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree
        s_out = ge.lib.to_sparse(out_strength, (num_vertices, num_labels),
                                 i_col='id', j_col='label', data_col='value',
                                 kind='csr').toarray()
        s_in = ge.lib.to_sparse(in_strength, (num_vertices, num_labels),
                                i_col='id', j_col='label', data_col='value',
                                kind='csr').toarray()

        model.expected_av_nn_strength(sdir='out', ndir='out', by_label=True)
        exp = np.dot(p_c, s_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_nn_s_out_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='out', by_label=True)
        exp = np.dot(p_c, s_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_nn_s_in_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='in', by_label=True)
        exp = np.dot(p_c.T, s_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0].reshape(3, 1)
        np.testing.assert_allclose(
            model.exp_av_in_nn_s_in_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out', ndir='in', by_label=True)
        exp = np.dot(p_c.T, s_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0].reshape(3, 1)
        np.testing.assert_allclose(
            model.exp_av_in_nn_s_out_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out-in', ndir='out-in',
                                      by_label=True)
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)),
                     s_out + s_in)
        exp[d != 0] = exp[d != 0] / d[d != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_in_nn_s_out_in_label, exp, rtol=1e-6)

    def test_av_nn_strength_label_multi_z(self):
        """ Test average nn strength."""
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        p_c = 1 - np.prod(1 - p_ref_lbl, axis=0)
        model.expected_degrees()
        d_out = model.exp_out_degree
        d_in = model.exp_in_degree
        s_out = ge.lib.to_sparse(out_strength, (num_vertices, num_labels),
                                 i_col='id', j_col='label', data_col='value',
                                 kind='csr').toarray()
        s_in = ge.lib.to_sparse(in_strength, (num_vertices, num_labels),
                                i_col='id', j_col='label', data_col='value',
                                kind='csr').toarray()

        model.expected_av_nn_strength(sdir='out', ndir='out', by_label=True)
        exp = np.dot(p_c, s_out)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_nn_s_out_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='out', by_label=True)
        exp = np.dot(p_c, s_in)
        exp[d_out != 0] = exp[d_out != 0] / d_out[d_out != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_nn_s_in_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='in', ndir='in', by_label=True)
        exp = np.dot(p_c.T, s_in)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0].reshape(3, 1)
        np.testing.assert_allclose(
            model.exp_av_in_nn_s_in_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out', ndir='in', by_label=True)
        exp = np.dot(p_c.T, s_out)
        exp[d_in != 0] = exp[d_in != 0] / d_in[d_in != 0].reshape(3, 1)
        np.testing.assert_allclose(
            model.exp_av_in_nn_s_out_label, exp, rtol=1e-6)

        model.expected_av_nn_strength(sdir='out-in', ndir='out-in',
                                      by_label=True)
        d = model.exp_degree
        exp = np.dot((1 - (1 - p_c)*(1 - p_c.T)),
                     s_out + s_in)
        exp[d != 0] = exp[d != 0] / d[d != 0].reshape(4, 1)
        np.testing.assert_allclose(
            model.exp_av_out_in_nn_s_out_in_label, exp, rtol=1e-6)

    def test_likelihood_single_z(self):
        """ Test likelihood code. """
        # Compute reference from p_ref
        p_log = p_ref.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.log(1 - p_ref)
        adj = np.zeros(p_ref.shape)
        adj[0, 0, 1] = 1
        adj[0, 1, 2] = 1
        adj[0, 3, 1] = 1
        adj[1, 2, 1] = 1
        adj[2, 3, 1] = 1
        adj[3, 1, 0] = 1

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k] != 0:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        # Construct model
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        assert np.abs(ref - model.log_likelihood(g)) < 1e-6

    def test_likelihood_multi_z(self):
        """ Test likelihood code. """
        # Compute reference from p_ref
        p_log = p_ref_lbl.copy()
        p_log[p_log != 0] = np.log(p_log[p_log != 0])
        np_log = np.log(1 - p_ref_lbl)
        adj = np.zeros(p_ref_lbl.shape)
        adj[0, 0, 1] = 1
        adj[0, 1, 2] = 1
        adj[0, 3, 1] = 1
        adj[1, 2, 1] = 1
        adj[2, 3, 1] = 1
        adj[3, 1, 0] = 1

        ref = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                for k in range(adj.shape[2]):
                    if adj[i, j, k] != 0:
                        ref += p_log[i, j, k]
                    else:
                        ref += np_log[i, j, k]

        # Construct model
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        assert np.abs(ref - model.log_likelihood(g)) < 1e-6

    def test_likelihood_error(self):
        """ Test likelihood code. """
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 0]])

        # Construct model
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        msg = 'Element 1 not a matrix'
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood([adj, 'dsf'])

        msg = re.escape('Passed adjacency matrix must have three '
                        'dimensions: (label, source, destination).')
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood(adj)

        msg = ('g input not a graph or list of adjacency matrices or '
               'numpy array.')
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood('dfsg')

        msg = re.escape('Number of passed layers (one per label) in adjacency '
                        'matrix is 1 instead of 4.')
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood([adj, ])

        msg = re.escape('Passed layer 0 adjacency matrix has shape (3, 4) '
                        'instead of (4, 4)')
        with pytest.raises(ValueError, match=msg):
            model.log_likelihood([adj, ]*4)


class TestFitnessModelSample():
    def test_sampling_single_z(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z)

        samples = 10000
        s_n_e = np.empty(samples)
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges
            assert np.all(sample.num_labels == num_labels)
            assert np.all(sample.num_vertices == num_vertices)

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges, atol=1e-1, rtol=0)

    def test_sampling_multi_z(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      param=z_label)

        samples = 10000
        s_n_e = np.empty((samples, num_labels))
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges_label
            assert np.all(sample.num_labels == num_labels)
            assert np.all(sample.num_vertices == num_vertices)

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges_label, atol=1e-1, rtol=0)
