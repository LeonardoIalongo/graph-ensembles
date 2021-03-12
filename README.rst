.. image:: https://travis-ci.com/LeonardoIalongo/graph-ensembles.svg?branch=master
    :target: https://travis-ci.com/LeonardoIalongo/graph-ensembles

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
Currently only the RandomGraph and StripeFitnessModel are fully implemented.
An example of how it can be used is the following. 
For more see the example notebooks in the examples folder.

.. code-block:: python

    import graph_ensembles as ge
    import pandas as pd

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

    # Initialize model
    model = ge.StripeFitnessModel(g)

    # Fit model parameters
    model.fit()

    # Sample from the ensemble
    model.sample()

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv env 
    source env/bin/activate 
    pip install -e '.[dev]'

For testing:

.. code-block:: bash

    pytest --cov

Credits
-------
This is a project by `Leonardo Niccolò Ialongo <https://datasciencephd.eu/students/leonardo-niccol%C3%B2-ialongo/>`_ and `Emiliano Marchese <https://www.imtlucca.it/en/emiliano.marchese/>`_, under 
the supervision of `Diego Garlaschelli <https://networks.imtlucca.it/members/diego>`_.

