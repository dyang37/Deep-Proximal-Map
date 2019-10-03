# Cython compile instructions

from distutils.core import setup
from Cython.Build import cythonize

# Use python setup.py build_ext --inplace
# to compile

setup(
  name = "icdapp",
  ext_modules = cythonize('*.pyx'),
)
