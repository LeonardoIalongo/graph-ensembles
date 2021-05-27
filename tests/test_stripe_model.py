""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd
import pytest

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
z = 1e-1
z_label = np.array([1.826529e-09, 3.060207e-07, 3.303774e-04, 1.011781e-03])


class TestStripeFitnessModelInit():
    def test_issubclass(self):
        """ Check that the stripe model is a graph ensemble."""
        model = ge.StripeFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        """ Check that the stripe model can be correctly initialized from
        a graph object.
        """
        model = ge.StripeFitnessModel(g)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)

        model = ge.StripeFitnessModel(g, per_label=False)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
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

    def test_model_init_z(self):
        """ Check that the stripe model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.z == z)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-5)

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.z == z_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
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

        msg = 'Number of vertices smaller then max id value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=2,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices must be a number.'
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

        msg = 'Number of labels must be a number.'
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

        msg = ("Out strength must be a rec array with columns: "
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

        msg = ("In strength must be a rec array with columns: "
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
        tmp.value[0] = 0
        with pytest.warn(UserWarning, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_out_strength(self):
        """ Test that an error is raised if out_strength contains negative
        values in either id, label or value.
        """
        tmp = out_strength.copy()

        tmp.label[4] = -tmp.label[4]
        msg = "Out strength labels must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.label[4] = -tmp.label[4]
        tmp.id[2] = -tmp.id[2]
        msg = "Out strength ids must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "Out strength values must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_in_strength(self):
        """ Test that an error is raised if in_strength contains negative
        values in either id, label or value.
        """
        tmp = in_strength.copy()

        tmp.label[1] = -tmp.label[1]
        msg = "In strength labels must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.label[1] = -tmp.label[1]
        tmp.id[2] = -tmp.id[2]
        msg = "In strength ids must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "In strength values must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
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

        msg = 'Either num_edges or z must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength)

        msg = ('Number of edges array does not have the number of'
               ' elements equal to the number of labels.')
        with pytest.raises(Exception, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=np.array([1, 2]))

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = ('z must be a number or an array of length equal to the number '
               'of labels.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z='three')

        msg = ('z array does not have the number of'
               ' elements equal to the number of labels.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=np.array([0, 1]))

        msg = 'z must contain only positive numbers.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=-1)
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=-z_label)


class TestFitnessModelFit():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.StripeFitnessModel(g)
        model.fit(method="newton")
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-5, rtol=0)

    def test_solver_invariant(self):
        """ Check that the newton solver is fitting the z parameters
        correctly for the invariant case. """
        model = ge.StripeFitnessModel(g, scale_invariant=True)
        model.fit(method="newton")
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-5, rtol=0)

    def test_solver_fixed_point(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.

        NOTE: currently very slow convergence for small graphs!
        """
        model = ge.StripeFitnessModel(g)
        model.fit(method="fixed-point", max_iter=200000, xtol=1e-5)
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-4, rtol=0)

    def test_sampling(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z)

        samples = 10000
        s_n_e = np.empty((samples, num_labels))
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges_label
            assert np.all(sample.num_labels == num_labels)
            assert np.all(sample.num_vertices == num_vertices)

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges, atol=1e-1, rtol=0)
