from setuptools import setup

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="utf-8") as history_file:
    history = history_file.read()

setup(
    name="graph-ensembles",
    author="Leonardo Niccolò Ialongo",
    author_email="leonardo.ialongo@gmail.com",
    python_requires=">=3.0",
    version="0.3.5",
    url="https://github.com/LeonardoIalongo/graph-ensembles",
    description=(
        "The graph ensemble package contains a set of methods to"
        " build fitness based graph ensembles from marginal"
        " information."
    ),
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    license="GNU General Public License v3",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    packages=[
        "graph_ensembles",
        "graph_ensembles.sparse",
        "graph_ensembles.sparse.models",
        "graph_ensembles.spark",
    ],
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22",
        "numba>=0.56",
        "scipy>=1.9",
        "pandas>=1.1",
        "networkx>=3.0",
    ],
)
