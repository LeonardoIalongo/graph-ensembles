""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge


def test_model_init():
    """ Check that a model can be correctly initialized with margins data."""
    data = None
    model = ge.VectorFitnessModel(data=data)
    assert model.data == data
