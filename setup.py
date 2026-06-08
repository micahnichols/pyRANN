# import subprocess
# from setuptools import setup, find_packages, Extension
# from setuptools.command.build_ext import build_ext
# import sys
# import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig
import pathlib
import pybind11

# ----------------------------------------------------------------
# Source configuration
# ----------------------------------------------------------------

SRC_DIR = pathlib.Path("c_src")
CPP_SOURCES = [str(p) for p in SRC_DIR.glob("*.cpp")]

# ----------------------------------------------------------------
# Compiler / linker flags (matching your Makefile)
# ----------------------------------------------------------------

extra_compile_args = [
    "-O2",
    "-fPIC",
    "-shared-libgcc",
    "-MMD",
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
        name="pyrann.calibration",  # matches TARGET := calibration$(EXT_SUFFIX)
        sources=CPP_SOURCES,
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

# setup(
#     name="calibration",
#     version="0.1",
#     author="Your Name",
#     description="Calibration module built with pybind11",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": build_ext},
#     zip_safe=False,
# )
class CustomBuildExt(build_ext):
    def run(self):
        os.chdir('c_src')
        # subprocess.check_call(['make'])
        os.chdir('../')
        # build_ext.run(self)
        # super().run()
        subprocess.run(['make -j'])

setup(
    name = 'pyrann',
    version = '1.0.0.a1',
    authors = ['Micah Nichols', 'Christopher Barrett'],
    description = 'Python package to generate atomic systems for use in DFT database for RANN interatomic potentials',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=['pyrann'],
    # package_dir={'': 'pyrann'},
    # package_data={
    #     'pyrann': ['*.so'],
    # },
    # include_package_data=True,
    # cmdclass={
    #     'build_ext': CustomBuildExt,
    # },
    install_requires = [
        'numpy',
        'scipy',
        ],
    zip_safe=False
    # packages = find_packages()
    )
