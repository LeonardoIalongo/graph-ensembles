""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd

# Define graph and its marginals to check computation
edges = pd.DataFrame({
    'src': [2, 3, 1, 4, 3],
    'dst': [1, 2, 4, 1, 1],
    'weight': [5, 3, 2, 1, 3]}, dtype=np.int8)

edges = edges.astype({'weight': 'float64'})

vertices = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'group': [1, 2, 3, 3]}, dtype=np.int8)

adjacency = np.array([[0, 0, 0, 2],
                      [5, 0, 0, 0],
                      [3, 3, 0, 0],
                      [1, 0, 0, 0]]
                     )

out_strength = np.array([[2, 0, 0],
                        [0, 5, 0],
                        [0, 0, 6],
                        [0, 0, 1]])

in_strength = np.array([[0, 5, 4],
                        [0, 0, 3],
                        [0, 0, 0],
                        [2, 0, 0]])

num_links = 5


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

    def test_probability_matrix(self):
        """ Check that the returned probability matrix is the correct one. """
        true_value = np.array([[0, 0, 0, 0.744985],
                              [0.948075, 0, 0, 0],
                              [0.946028, 0.929309, 0, 0],
                              [0.744985, 0.686619, 0, 0]])

        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        np.testing.assert_allclose(model.probability_matrix.toarray(),
                                   true_value,
                                   rtol=1e-6)

    def test_expected_num_links(self):
        model = ge.VectorFitnessModel(out_strength,
                                      in_strength,
                                      num_links)
        exp_num_links = model.probability_matrix.sum()
        assert exp_num_links == num_links
