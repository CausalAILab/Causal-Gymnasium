from setuptools import setup, find_packages

setup(
    name="causal_gym",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ale-py>=0.11.1",
        "gymnasium==1.1.1",
        "highway-env>=1.10.1",
        "matplotlib>=3.10.3",
        "minigrid>=3.0.0",
        "networkx>=3.5",
        "numpy>=2.2.6",
        "opencv-python>=4.11.0.86",
        "Pillow>=11.2.1",
        "pygame>=2.6.1",
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "multiprocess>=0.70.16",  # Retained from original
        "Box2D>=2.3.10",
    ],
    extras_require={
        'dev': [
            'ipykernel',  # good for testing w/ notebooks
            'imageio',    # for creating GIFs
        ]
    }
)
