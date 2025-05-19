from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension("gobangboard",
        sources=["gobangboard.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=[])]

setup(
    name = "gobangboard",
    ext_modules = cythonize(extensions)
)

# python setup.py build_ext --inplace