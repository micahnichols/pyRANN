# import subprocess
# from setuptools import setup, find_packages, Extension
from setuptools import setup, Extension
import sys
import sysconfig
from pathlib import Path
import pybind11

# ----------------------------------------------------------------
# Source configuration
# ----------------------------------------------------------------

SRC_DIR = Path('c_src').resolve()
CPP_SOURCES = [str(p) for p in SRC_DIR.glob("*.cpp")]
pybind_str = str(Path('pyrann') / 'pybind_calibration.cpp')
CPP_SOURCES = [pybind_str] + CPP_SOURCES

# ----------------------------------------------------------------
# Compiler / linker flags (matching your Makefile)
# ----------------------------------------------------------------

extra_compile_args = [
    "-O3",
    "-fPIC",
    "-shared-libgcc",
    "-fopenmp",
]

extra_link_args = [
    "-shared",
    "-fopenmp",
]

# ----------------------------------------------------------------
# Extension definition
# ----------------------------------------------------------------

ext_modules = [
    Extension(
        name="pyrann.calibration",
        sources=CPP_SOURCES,
        include_dirs=[
            pybind11.get_include(),
            SRC_DIR,
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

setup(
    name = 'pyrann',
    version = '1.0.0.a1',
    author = 'Micah Nichols, Christopher Barrett',
    description = 'Python package to generate atomic systems for use in DFT database for RANN interatomic potentials',
    ext_modules = ext_modules,
    packages=['pyrann'],
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'bokeh',
        'matplotlib',
        'colorcet',
        'quippy-ase',
        'ase',
        'umap-learn',
        ],
    extras_require = {
        'mtp': ['pyrann-mtp @ file:./pyrann-mtp'],
        'all': ['pyrann-mtp @ file:./pyrann-mtp'],
        },
    zip_safe = False
    )
