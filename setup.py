# pybind11 build helper (keeps a single C++ source file as the extension)
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cascade_sim",
        ["cascade_engine.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="interdependent-percolation",
    version="0.1.0",
    description="Interdependent percolation on ER graphs: Python + pybind11 C++ cascade engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    py_modules=["site_percolation", "plot_cascade"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "networkx>=3.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": [
            "pybind11>=2.12",
            "pytest>=7.0",
            "ruff>=0.4",
        ],
    },
)
