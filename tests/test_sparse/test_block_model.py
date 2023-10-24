""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd
import pytest
import re

v = pd.DataFrame([['ING', 'NL'],
                 ['ABN', 'NL'],
                 ['UNI', 'IT'],
                 ['BNL', 'IT']],
                 columns=['name', 'country'])

e_var = pd.DataFrame([['ING', 'ABN', 1e6],
                     ['ABN', 'UNI', 2.3e7],
                     ['BNL', 'UNI', 7e5],
                     ['UNI', 'BNL', 3e3]],
                     columns=['creditor', 'debtor', 'value'])

e = pd.DataFrame([['ING', 'ABN', 1e6],
                  ['ABN', 'UNI', 2.3e7],
                  ['BNL', 'UNI', 7e5],
                  ['ABN', 'BNL', 6e4],
                  ['ING', 'BNL', 5e2],
                  ['UNI', 'BNL', 3e3]],
                 columns=['creditor', 'debtor', 'value'])

g = ge.Graph(v, e, v_id='name', src='creditor', dst='debtor',
             weight='value', v_group='country')

g_var = ge.Graph(v, e_var, v_id='name', src='creditor', dst='debtor',
                 weight='value', v_group='country')

# Define graph marginals to check computation
out_strength = np.rec.array([(0, 0, 1e6),
                             (0, 1, 5e2),
                             (1, 1, 2.3e7 + 6e4),
                             (2, 1, 3e3),
                             (3, 1, 7e5)],
                            dtype=[('id', np.uint8),
                                   ('group', np.uint8),
                                   ('value', np.float64)])

in_strength = np.rec.array([(1, 0, 1e6),
                            (2, 0, 2.3e7),
                            (2, 1, 7e5),
                            (3, 0, 5e2 + 6e4),
                            (3, 1, 3e3)],
                           dtype=[('id', np.uint8),
                                  ('group', np.uint8),
                                  ('value', np.float64)])

num_vertices = 4
num_edges = 6
num_edges_var = 4
num_groups = 2
group_dict = {1: 0, 2: 1, 0: 0, 3: 1}
wrong_group_dict = {0: 0, 1: 0}
z = 6.08040786e-08
z_var = 27.843357146819287


class TestBlockFitnessModelInit():
    def test_issubclass(self):
        """ Check that the block model is a graph ensemble."""
        model = ge.BlockFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        """ Check that the block model can be correctly initialized from
        parameters directly.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     num_edges=num_edges)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.group_dict == np.array([0, 0, 1, 1]))
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_g(self):
        """ Check that the block model can be correctly initialized from a
        graph.
        """
        model = ge.BlockFitnessModel(g)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.group_dict == np.array([0, 0, 1, 1]))
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
        """ Check that the block model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     param=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.param == z)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the model raises exceptions for wrong inputs."""
        msg = 'First argument passed must be a WeightedGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel('df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.BlockFitnessModel(g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_nodes=num_vertices,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of vertices smaller than max id value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=2,
                                 num_groups=num_groups,
                                 group_dict=wrong_group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=np.array([1, 2]),
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=-3,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_wrong_num_groups(self):
        """ Check that wrong initialization of num_groups results in an error.
        """
        msg = 'Number of groups not set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of groups smaller than max group value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=1,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of groups must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=np.array([1, 2]),
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Number of groups must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=-5,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_wrong_group_dict(self):
        """ Check correct errors if group_dict is not right.
        """
        msg = 'Group dictionary not set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Group_dict must have one element for each vertex.'
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=wrong_group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'Group dictionary must be a dict or an array.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=[0, 1, 2],
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of strengths results in an error.
        """
        msg = 'out_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = 'in_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 num_edges=num_edges)

        msg = re.escape("Out strength must be a rec array with columns: "
                        "('id', 'group', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=1,
                                 in_strength=in_strength,
                                 num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength.value,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = re.escape("In strength must be a rec array with columns: "
                        "('id', 'group', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=2,
                                 num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength.value,
                                 num_edges=num_edges)

        msg = "Sums of strengths do not match."
        tmp = out_strength.copy()
        tmp.value[0] = tmp.value[0] + 1
        with pytest.raises(AssertionError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=tmp,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        msg = "Storing zeros in the strengths leads to inefficient code."
        tmp = out_strength.copy()
        tmp.resize(len(tmp) + 1)
        tmp[-1] = ((1, 1, 0))
        with pytest.warns(UserWarning, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=tmp,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_negative_out_strength(self):
        """ Test that an error is raised if out_strength contains negative
        values in either id, label or value.
        """
        tmp = out_strength.copy().astype([('id', np.int8),
                                          ('group', np.int8),
                                          ('value', np.float64)])

        tmp.group[1] = -1
        msg = "Out strength groups must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=tmp,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        tmp.group[1] = out_strength.group[1]
        tmp.id[2] = -tmp.id[2]
        msg = "Out strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=tmp,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "Out strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=tmp,
                                 in_strength=in_strength,
                                 num_edges=num_edges)

    def test_negative_in_strength(self):
        """ Test that an error is raised if in_strength contains negative
        values in either id, label or value.
        """
        tmp = in_strength.copy().astype([('id', np.int8),
                                         ('group', np.int8),
                                         ('value', np.float64)])

        tmp.group[2] = -tmp.group[2]
        msg = "In strength groups must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=tmp,
                                 num_edges=num_edges)

        tmp.group[2] = -tmp.group[2]
        tmp.id[2] = -tmp.id[2]
        msg = "In strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=tmp,
                                 num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "In strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=tmp,
                                 num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = ('Number of edges must be a number.')
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges='f')
        with pytest.raises(Exception, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=np.array([1, 2]))

        msg = 'Number of edges must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 num_edges=-324)

        msg = 'Either num_edges or param must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = 'Parameter must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 param=np.array([0, 1]))
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 param='l')

        msg = 'Parameter must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.BlockFitnessModel(num_vertices=num_vertices,
                                 num_groups=num_groups,
                                 group_dict=group_dict,
                                 out_strength=out_strength,
                                 in_strength=in_strength,
                                 param=-1)

class TestBlockFitnessModelFit():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.BlockFitnessModel(g)
        model.fit(method="newton")
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        assert np.isclose(z, model.param[0], atol=1e-12, rtol=1e-6)

    def test_solver_newton_var(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.BlockFitnessModel(g_var)
        model.fit(method="newton")
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges_var, model.exp_num_edges,
                                   atol=1e-5, rtol=0)

    def test_solver_invariant(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.BlockFitnessModel(g, scale_invariant=True)
        model.fit()
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)

    def test_solver_invariant_var(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.BlockFitnessModel(g_var, scale_invariant=True)
        model.fit()
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges_var, model.exp_num_edges,
                                   atol=1e-5, rtol=0)

    def test_solver_fixed_point(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.BlockFitnessModel(g)
        model.fit(method="fixed-point", max_iter=200000, xtol=1e-5)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-4, rtol=0)

    def test_solver_fixed_point_var(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.BlockFitnessModel(g_var)
        model.fit(method="fixed-point", max_iter=200000, xtol=1e-5)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges_var, model.exp_num_edges,
                                   atol=1e-4, rtol=0)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.BlockFitnessModel(g)
        model.fit(x0=1e-14)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param[0], atol=0, rtol=1e-4)

    def test_solver_with_wrong_init(self):
        """ Check that it raises an error with a negative initial condition.
        """
        msg = 'x0 must be positive.'
        with pytest.raises(ValueError, match=msg):
            model = ge.BlockFitnessModel(g)
            model.fit(x0=-1)

        msg = 'x0 must be a number.'
        with pytest.raises(ValueError, match=msg):
            model = ge.BlockFitnessModel(g)
            model.fit(x0='lef')
        with pytest.raises(ValueError, match=msg):
            model = ge.BlockFitnessModel(g)
            model.fit(x0=np.array([0, 1]))

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.BlockFitnessModel(g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")

    def test_method_incompatibility(self):
        """ Check that an error is raised when trying to use the wrong method.
        """
        model = ge.BlockFitnessModel(g, scale_invariant=True)
        msg = ('Fixed point solver not supported for scale '
               'invariant functional.')
        with pytest.raises(Exception, match=msg):
            model.fit(method="fixed-point", max_iter=100, xtol=1e-5)


class TestFitnessModelMeasures():
    def test_exp_n_edges(self):
        """ Check expected edges is correct. """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     param=z)
        model.expected_num_edges()
        np.testing.assert_allclose(model.exp_num_edges,
                                   num_edges,
                                   rtol=1e-5)

    def test_exp_out_degree(self):
        """ Check expected d_out is correct. """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     param=z)
        model.expected_degrees()
        d_out = model.exp_out_degree
        np.testing.assert_allclose(
            d_out, np.array([2, 1, 0, 0]),
            rtol=1e-5)

    def test_exp_in_degree(self):
        """ Check expected d_in is correct. """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     param=z)
        model.expected_degrees()
        d_in = model.exp_in_degree
        np.testing.assert_allclose(
            d_in, np.array([0, 0, 2, 2]), rtol=1e-5)


class TestBlockFitnessModelSample():
    def test_sampling(self):
        """ Check that properties of the sample correspond to the ensemble.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     param=z)

        samples = 10000
        avg = 0
        for i in range(samples):
            sample = model.sample()
            avg += sample.num_edges
            assert sample.num_groups == num_groups
            assert sample.num_vertices == num_vertices

        avg = avg / samples
        assert np.isclose(avg, num_edges, atol=1e-1, rtol=0)
