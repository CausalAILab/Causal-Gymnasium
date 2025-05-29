from setuptools import setup

setup(
    name="causal_gym",
    version="1.0.0",
    install_requires=[
        "gymnasium>=0.29.1", 
        "pygame>=2.5.2", 
        "multiprocess>=0.70.16",
        "highway-env>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5,<3.10",
    ],
)
