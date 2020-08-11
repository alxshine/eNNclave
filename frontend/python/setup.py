from distutils.core import setup, Extension
from os.path import join
import os

setup(name='eNNclave',
      version='1.0',
      author="Alexander Schlögl",
      description="Python frontend for eNNclave",
      author_email="alexander.schloegl@uibk.ac.at",
      py_modules=['ennclave'],
      ext_modules=[Extension('frontend_python', ['interoperability/frontend_python.c'], libraries=['dl'],
                             include_dirs=[join(os.getenv('ENNCLAVE_HOME'), 'inc')])],
      install_requires=['jinja2', 'tensorflow-gpu', 'numpy', 'invoke']
      )
