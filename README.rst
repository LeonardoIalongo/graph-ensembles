=================
Graph ensembles
=================

The graph ensemble package contains a set of methods to build fitness based 
graph ensembles from marginal information. These methods can be used to build 
randomized ensembles preserving the marginal information provided. 

* Free software: GNU General Public License v3
* Documentation: https://graph-ensembles.readthedocs.io.


Installation
------------
Install using:

.. code-block:: python

   pip install graph_ensembles

Usage
-----
Currently only the StripeFitnessModel is fully implemented. An example of how 
it can be used is the following. For more see the example notebooks in the 
examples folder.

.. code-block:: python

   # Define graph marginals
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

    # Initialize model
    model = ge.StripeFitnessModel(out_strength, in_strength, num_links)

    # Fit model parameters
    model.fit()

    # Return probability matrix 
    prob_mat = model.probability_matrix

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv venv 
    source venv/bin/activate 
    pip install -e '.[dev]'

For testing:

.. code-block:: bash

    pytest --cov

Credits
-------
This is a project by `Leonardo Niccolò Ialongo <https://datasciencephd.eu/students/leonardo-niccol%C3%B2-ialongo/>`_ and `Emiliano Marchese <https://www.imtlucca.it/en/emiliano.marchese/>`_, under 
the supervision of `Diego Garlaschelli <https://networks.imtlucca.it/members/diego>`_.

