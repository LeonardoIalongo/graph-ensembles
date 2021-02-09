""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pytest

# Define graph marginals to check computation
out_strength = np.array([[0, 0, 2],
                        [1, 1, 5],
                        [2, 2, 6],
                        [3, 2, 1]])

in_strength = np.array([[0, 1, 5],
                        [0, 2, 4],
                        [1, 2, 3],
                        [3, 0, 2]])

num_nodes = 4
num_links = np.array([1, 1, 3])
num_links_tot = 5
num_groups = 3


class TestStripeFitnessModel():
    def test_issubclass(self):
        """ Check that the stripe model is a graph model."""
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        assert isinstance(model, ge.GraphModel)

    def test_model_init_single(self):
        """ Check that the stripe model can be correctly initialized with a
        single number of links.
        """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_links == num_links_tot)

    def test_model_init_multi(self):
        """ Check that the stripe model can be correctly initialized with
        an array of number of links.
        """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_links == num_links)

    def test_model_wrong_init(self):
        """ Check that the stripe model raises exceptions for wrong inputs."""
        with pytest.raises(Exception) as e_info:
            ge.StripeFitnessModel(out_strength,
                                  in_strength,
                                  np.array([1, 2]))
        msg = ('Number of links array does not have the number of'
               ' elements equal to the number of labels.')
        assert e_info.value.args[0] == msg

        with pytest.raises(Exception) as e_info:
            ge.StripeFitnessModel(out_strength,
                                  in_strength,
                                  [1, 2])
        assert e_info.value.args[0] == 'Number of links is not a number.'

    def test_solver_single_LS(self):
        """ Check that the scipy least-squares solver is fitting the parameter z correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        model.fit(z0=0, method="least-squares")
        np.testing.assert_allclose(model.z,
                                   0.730334,
                                   rtol=1e-6)

    def test_solver_single_fixed_point(self):
        """ Check that the fixed-point solver is fitting the parameter z correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        model.fit(z0=0.1, method="fixed-point")
        np.testing.assert_allclose(model.z,
                                   0.730334,
                                   rtol=1e-6)

    def test_solver_single_newton(self):
        """ Check that the newton solver is fitting the parameter z
        correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        model.fit(z0=0.1, method="newton")
        np.testing.assert_allclose(model.z,
                                   0.730334,
                                   rtol=1e-6)

    def test_solver_multi_LS(self):
        """ Check that the scipy least-squares solver is fitting the
        parameter z correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        model.fit(z0=[0, 0, 0], method="least-squares")
        np.testing.assert_allclose(model.z,
                                   [1.006638e+08, 1.610613e+07, 4.346469e-01],
                                   rtol=1e-6)

    def test_solver_multi_fixed_point(self):
        """ Check that the fixed-point solver is fitting the parameter z correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        model.fit(z0=np.array([0.1, 0.1, 0.1]), method="fixed-point", max_steps=10000)
        np.testing.assert_allclose(model.probability_matrix.sum(),
                                   model.num_links.sum(),
                                   rtol=1e-3)

    def test_solver_multi_newton(self):
        """ Check that the newton solver is fitting the parameter z correctly. """
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        model.fit(z0=np.array([0.1, 0.1, 0.1]), method="newton")
        np.testing.assert_allclose(model.probability_matrix.sum(),
                                   model.num_links.sum(),
                                   rtol=1e-4)

    def test_probability_array_single(self):
        """ Check that the returned probability array is correct. """
        true_value = np.zeros((num_nodes, num_nodes, num_groups),
                              dtype=np.float64)

        true_value[0, 3, 0] = 0.744985
        true_value[1, 0, 1] = 0.948075
        true_value[2, 0, 2] = 0.946028
        true_value[2, 1, 2] = 0.929309
        true_value[3, 0, 2] = 0.744985
        true_value[3, 1, 2] = 0.686619

        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        np.testing.assert_allclose(model.probability_array,
                                   true_value,
                                   rtol=1e-6)

    def test_probability_array_multi(self):
        """ Check that the returned probability array is correct. """
        true_value = np.zeros((num_nodes, num_nodes, num_groups),
                              dtype=np.float64)
        true_value[0, 3, 0] = 1.
        true_value[1, 0, 1] = 1.
        true_value[2, 0, 2] = 0.912523
        true_value[2, 1, 2] = 0.886668
        true_value[3, 0, 2] = 0.634848
        true_value[3, 1, 2] = 0.565961

        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        np.testing.assert_allclose(model.probability_array,
                                   true_value,
                                   rtol=1e-6)

    def test_probability_matrix_single(self):
        """ Check that the returned probability matrix is the correct one. """
        true_value = np.array([[0., 0., 0., 0.744985],
                              [0.948075, 0., 0., 0.],
                              [0.946028, 0.929309, 0., 0.],
                              [0.744985, 0.686619, 0., 0.]])

        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        np.testing.assert_allclose(model.probability_matrix,
                                   true_value,
                                   rtol=1e-6)

    def test_probability_matrix_multi(self):
        """ Check that the returned probability matrix is the correct one. """
        true_value = np.array([[0., 0., 0., 1.],
                              [1., 0., 0., 0.],
                              [0.912523, 0.886668, 0., 0.],
                              [0.634848, 0.565961, 0., 0.]])

        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        np.testing.assert_allclose(model.probability_matrix,
                                   true_value,
                                   rtol=1e-6)

    def test_expected_num_links_single(self):
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links_tot)
        exp_num_links = model.probability_matrix.sum()
        np.testing.assert_allclose(num_links_tot,
                                   exp_num_links,
                                   rtol=1e-6)

    def test_expected_num_links_multi(self):
        model = ge.StripeFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        exp_num_links = np.sum(model.probability_array, axis=(0, 1))
        np.testing.assert_allclose(num_links,
                                   exp_num_links,
                                   rtol=1e-6)


class TestHelper():
    def test_random_edge_list_noself(self):
        """ Check that the randomly generated edge list is consistent."""
        N = 10
        L = 50
        W = 30
        G = 5
        edges = ge.helper.random_edge_list(N, L, W, G=G, self_loops=False)
        assert np.all([i != j for (i, j) in edges[['src', 'dst']]])
