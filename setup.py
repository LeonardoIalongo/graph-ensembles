from setuptools import setup

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf-8') as history_file:
    history = history_file.read()

setup(
    name='graph-ensembles',
    author="Leonardo Niccolò Ialongo",
    author_email='leonardo.ialongo@gmail.com',
    python_requires='>=3.0',
    version='0.2.0',
    url='https://github.com/LeonardoIalongo/graph-ensembles',
    description=("The graph ensemble package contains a set of methods to"
                 " build fitness based graph ensembles from marginal"
                 " information."),
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    license="GNU General Public License v3",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    packages=['graph_ensembles', 'graph_ensembles.methods'],
    package_dir={'': 'src'},
    install_requires=["numpy>=1.15",
                      "numba>=0.53",
                      "scipy>=1.0",
                      "pandas>=1.1",
                      "networkx>=1.0"
                      ],
    extras_require={
        "dev": ["pytest==6.0.1",
                "coverage==5.2.1",
                "pytest-cov==2.10.1",
                "flake8==3.8.3",
                "wheel==0.35.1",
                "matplotlib==3.3.2",
                "check-manifest==0.44",
                "setuptools==47.1.0",
                "twine==3.2.0",
                "tox==3.20.1"],
        },
    )
