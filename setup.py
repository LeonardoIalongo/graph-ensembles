from setuptools import setup

setup(
    name='graph_ensembles',
    version='0.0.1',
    py_modules=['graph_ensembles'],
    package_dir={'': 'src'},
    install_requires=["numpy>=1.19",
                      "scipy>=1.5"],
    extras_require={
        "dev": ["pytest==6.0.1",
                "coverage==5.2.1",
                "pytest-cov==2.10.1",
                "flake8==3.8.3",
                "wheel==0.35.1",
		"matplotlib==3.3.2"],
        },
    )
