from setuptools import setup, Extension
from pathlib import Path
import pybind11

# ----------------------------------------------------------------
# Source configuration
# ----------------------------------------------------------------

SRC_DIR = Path('mlip-3-main') / 'src'
CPP_SOURCES = [str(i)
               for i in SRC_DIR.rglob('*.cpp')
               if i.name not in ('main.cpp', 'mtp_bindings.cpp')
               ]
CPP_SOURCES = ['mtp_bindings.cpp'] + CPP_SOURCES

# ----------------------------------------------------------------
# Compiler / linker flags
# ----------------------------------------------------------------

extra_compile_args = [
    "-O3",
    "-shared",
    "-fPIC",
]

extra_link_args = [
    "-lopenblas",
]

# ----------------------------------------------------------------
# Extension Definition
# ----------------------------------------------------------------

ext_modules = [
    Extension(
        name='pyrann_mtp.mtp_bindings',
        sources=CPP_SOURCES,
        include_dirs=[
            pybind11.get_include(),
            SRC_DIR,
        ],
        language='c++',
        libraries=['openblas'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

setup(
    name = 'pyrann_mtp',
    version = '1.0.0.a1',
    description = 'Pybind11 bindings for mlip-3 (MTP)',
    ext_modules=ext_modules,
    zip_safe=False,
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
        ]
)
