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
num_edges = np.array([3, 1, 1, 1])
num_labels = 4
z = np.array([1.826529e-09, 3.060207e-07, 3.303774e-04, 1.011781e-03])


class TestStripeFitnessModel():
    def test_issubclass(self):
        """ Check that the stripe model is a graph ensemble."""
        model = ge.StripeFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

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
                                   rtol=1e-6)

    def test_model_wrong_init(self):
        """ Check that the stripe model raises exceptions for wrong inputs."""
        with pytest.raises(Exception) as e_info:
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=np.array([1, 2]))

        msg = ('Number of edges array does not have the number of'
               ' elements equal to the number of labels.')
        assert e_info.value.args[0] == msg

    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.StripeFitnessModel(g)
        model.fit(method="newton")
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-8, rtol=0)

    def test_solver_fixed_point(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.

        NOTE: currently very slow convergence!
        """
        model = ge.StripeFitnessModel(g)
        model.fit(method="fixed-point", max_iter=100000, xtol=1e-5)
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
