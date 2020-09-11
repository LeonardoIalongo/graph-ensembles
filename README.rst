Network ensembles
=================


Development
-----------
Please work on a feature branch and merge back to development
branch when ready using::

    git merge --no-ff myfeature

To build a development environment run::

    python3 -m venv venv 
    source venv/bin/activate 
    pip install -e '.[dev]'

For testing::

    pytest` or `pytest --cov
