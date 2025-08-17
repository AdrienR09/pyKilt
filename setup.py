from setuptools import setup, find_packages

setup(
    name="pyKilt",
    version="0.1.0",
    author="Adrien Rousseau",
    description="Kinetic Inverse-Laplace Toolbox for MEM fitting",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "jaxopt",
        "data-analysis",
        "lmfit",
        "seaborn",
        "jax",
    ],
)
