from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension("goboard",
        sources=["goboard.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17",],
        extra_link_args=[])]

setup(
    name = "goboard",
    ext_modules = cythonize(extensions)
)

# python setup.py build_ext --inplace