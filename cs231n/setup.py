# encoding=utf8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys
try:
	reload(sys)
	sys.setdefaultencoding('utf-8')
except NameError:
	pass
extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
