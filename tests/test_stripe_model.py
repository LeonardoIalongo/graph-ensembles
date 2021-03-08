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
out_strength = np.array([(0, 0, 2),
                         (1, 1, 5),
                         (2, 2, 6),
                         (2, 3, 1)],
                        dtype=[('label', 'u1'),
                               ('id', 'u1'),
                               ('value', 'f8')]).view(np.recarray)

in_strength = np.array([(0, 3, 2),
                        (1, 0, 5),
                        (2, 0, 4),
                        (2, 1, 3)],
                       dtype=[('label', 'u1'),
                              ('id', 'u1'),
                              ('value', 'f8')]).view(np.recarray)

num_vertices = 4
num_edges = np.array([1, 1, 3])
num_edges_tot = 5
num_labels = 3
z = np.log(np.array([1.006638e+08, 1.610613e+07, 4.346469e-01]))


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

    # def test_solver_single_LS(self):
    #     """ Check that the scipy least-squares solver is fitting the parameter z correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     model.fit(z0=0, method="least-squares")
    #     np.testing.assert_allclose(model.z,
    #                                0.730334,
    #                                rtol=1e-6)

    # def test_solver_single_fixed_point(self):
    #     """ Check that the fixed-point solver is fitting the parameter z correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     model.fit(z0=0.1, method="fixed-point")
    #     np.testing.assert_allclose(model.z,
    #                                0.730334,
    #                                rtol=1e-6)

    # def test_solver_single_newton(self):
    #     """ Check that the newton solver is fitting the parameter z
    #     correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     model.fit(z0=0.1, method="newton")
    #     np.testing.assert_allclose(model.z,
    #                                0.730334,
    #                                rtol=1e-6)

    # def test_solver_multi_LS(self):
    #     """ Check that the scipy least-squares solver is fitting the
    #     parameter z correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     model.fit(z0=[0, 0, 0], method="least-squares")
    #     np.testing.assert_allclose(model.z,
    #                                [1.006638e+08, 1.610613e+07, 4.346469e-01],
    #                                rtol=1e-6)

    # def test_solver_multi_fixed_point(self):
    #     """ Check that the fixed-point solver is fitting the parameter z correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     model.fit(z0=np.array([0.1, 0.1, 0.1]), method="fixed-point", max_steps=10000)
    #     np.testing.assert_allclose(model.probability_matrix.sum(),
    #                                model.num_links.sum(),
    #                                rtol=1e-3)

    # def test_solver_multi_newton(self):
    #     """ Check that the newton solver is fitting the parameter z correctly. """
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     model.fit(z0=np.array([0.1, 0.1, 0.1]), method="newton")
    #     np.testing.assert_allclose(model.probability_matrix.sum(),
    #                                model.num_links.sum(),
    #                                rtol=1e-4)

    # def test_probability_array_single(self):
    #     """ Check that the returned probability array is correct. """
    #     true_value = np.zeros((num_nodes, num_nodes, num_groups),
    #                           dtype=np.float64)

    #     true_value[0, 3, 0] = 0.744985
    #     true_value[1, 0, 1] = 0.948075
    #     true_value[2, 0, 2] = 0.946028
    #     true_value[2, 1, 2] = 0.929309
    #     true_value[3, 0, 2] = 0.744985
    #     true_value[3, 1, 2] = 0.686619

    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     np.testing.assert_allclose(model.probability_array,
    #                                true_value,
    #                                rtol=1e-6)

    # def test_probability_array_multi(self):
    #     """ Check that the returned probability array is correct. """
    #     true_value = np.zeros((num_nodes, num_nodes, num_groups),
    #                           dtype=np.float64)
    #     true_value[0, 3, 0] = 1.
    #     true_value[1, 0, 1] = 1.
    #     true_value[2, 0, 2] = 0.912523
    #     true_value[2, 1, 2] = 0.886668
    #     true_value[3, 0, 2] = 0.634848
    #     true_value[3, 1, 2] = 0.565961

    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     np.testing.assert_allclose(model.probability_array,
    #                                true_value,
    #                                rtol=1e-6)

    # def test_probability_matrix_single(self):
    #     """ Check that the returned probability matrix is the correct one. """
    #     true_value = np.array([[0., 0., 0., 0.744985],
    #                           [0.948075, 0., 0., 0.],
    #                           [0.946028, 0.929309, 0., 0.],
    #                           [0.744985, 0.686619, 0., 0.]])

    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     np.testing.assert_allclose(model.probability_matrix,
    #                                true_value,
    #                                rtol=1e-6)

    # def test_probability_matrix_multi(self):
    #     """ Check that the returned probability matrix is the correct one. """
    #     true_value = np.array([[0., 0., 0., 1.],
    #                           [1., 0., 0., 0.],
    #                           [0.912523, 0.886668, 0., 0.],
    #                           [0.634848, 0.565961, 0., 0.]])

    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     np.testing.assert_allclose(model.probability_matrix,
    #                                true_value,
    #                                rtol=1e-6)

    # def test_expected_num_links_single(self):
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links_tot)
    #     exp_num_links = model.probability_matrix.sum()
    #     np.testing.assert_allclose(num_links_tot,
    #                                exp_num_links,
    #                                rtol=1e-6)

    # def test_expected_num_links_multi(self):
    #     model = ge.StripeFitnessModel(out_strength,
    #                                   in_strength,
    #                                   num_links)
    #     exp_num_links = np.sum(model.probability_array, axis=(0, 1))
    #     np.testing.assert_allclose(num_links,
    #                                exp_num_links,
    #                                rtol=1e-6)
