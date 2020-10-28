from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

# define an extension that will be cythonized and compiled
# ext = Extension(name="hello", sources=["hello.pyx"])
# setup(ext_modules=cythonize(ext))
setup(ext_modules=cythonize('dsGameSolver/homCreate_ct.pyx'), include_path=[np.get_include()])
