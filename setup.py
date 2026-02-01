from setuptools import setup, find_packages

setup(
    name="causal_gym",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium[all]>=0.29.1", 
        "pygame>=2.5.2", 
        "multiprocess>=0.70.16",
        "highway-env>=1.4.0",
        "numpy>=1.21.0,<=1.26.4",
        "matplotlib>=3.5,<3.10",
        "pydash>=5.1.0",
        "networkx>=2.6.3",
        "toposort>=1.7",
        "pydot>=1.4.2",
        "toposort",
        "pydash",
        "box2d-py",
        "torchvision==0.25.0",
        "ogbench==1.2.1",
    ],
    include_package_data=True,
)