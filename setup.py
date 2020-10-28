from setuptools import setup

with open('README.rst', 'r') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', 'r') as history_file:
    history = history_file.read()

setup(
    name='graph_ensembles',
    author="Leonardo Niccolò Ialongo",
    author_email='leonardo.ialongo@gmail.com',
    python_requires='>=3.0',
    version='0.0.1',
    url='https://github.com/LeonardoIalongo/graph_ensembles',
    description=("The graph ensemble package contains a set of methods to"
                 " build fitness based graph ensembles from marginal"
                 " information."),
    long_description=readme + '\n\n' + history,
    license="GNU General Public License v3",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    packages=['graph_ensembles'],
    package_dir={'': 'src'},
    install_requires=["numpy>=1.19",
                      "scipy>=1.5",
                      "numba>=0.51"
                      ],
    extras_require={
        "dev": ["pytest==6.0.1",
                "coverage==5.2.1",
                "pytest-cov==2.10.1",
                "flake8==3.8.3",
                "wheel==0.35.1",
                "matplotlib==3.3.2",
                "networkx==2.5",
                "check-manifest==0.44"],
        },
    )
