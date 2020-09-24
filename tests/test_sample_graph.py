""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np

# Define graph marginals to check computation
out_strength = np.array([[2, 0, 0],
                        [0, 5, 0],
                        [0, 0, 6],
                        [0, 0, 1]])

in_strength = np.array([[0, 5, 4],
                        [0, 0, 3],
                        [0, 0, 0],
                        [2, 0, 0]])

num_nodes = 4
num_links = 5
num_groups = 3


class TestVectorFitnessModel():
    def test_issubclass(self):
        """ Check that the vfm is a graph model."""
        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        assert isinstance(model, ge.GraphModel)

    def test_model_init(self):
        """ Check that the vfm can be correctly initialized with margins data.
        """
        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert model.num_links == num_links

    def test_solver(self):
        """ Check that the solver is fitting the parameter z correctly. """
        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        model.solve(z0=1)
        np.testing.assert_almost_equal(model.z, 0.730334, decimal=6)

    def test_probability_array(self):
        """ Check that the returned probability array is correct. """
        true_value = np.zeros((num_nodes, num_nodes, num_groups),
                              dtype=np.float64)
        true_value[0, 3, 0] = 0.744985
        true_value[1, 0, 1] = 0.948075
        true_value[2, 0, 2] = 0.946028
        true_value[2, 1, 2] = 0.929309
        true_value[3, 0, 2] = 0.744985
        true_value[3, 1, 2] = 0.686619

        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        np.testing.assert_allclose(model.probability_array,
                                   true_value,
                                   rtol=1e-6)

    def test_probability_matrix(self):
        """ Check that the returned probability matrix is the correct one. """
        true_value = np.array([[0, 0, 0, 0.744985],
                              [0.948075, 0, 0, 0],
                              [0.946028, 0.929309, 0, 0],
                              [0.744985, 0.686619, 0, 0]])

        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        np.testing.assert_allclose(model.probability_matrix,
                                   true_value,
                                   rtol=1e-6)

    def test_expected_num_links(self):
        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        exp_num_links = model.probability_matrix.sum()
        assert exp_num_links == num_links
