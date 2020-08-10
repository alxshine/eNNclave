from distutils.core import setup, Extension

setup(name='eNNclave',
      py_modules=['ennclave'],
      ext_modules=[Extension('frontend_python', ['interoperability/frontend_python.c'], libraries=['dl'],
                             include_dirs=['../../inc'])],
      install_requires=['jinja2', 'tensorflow', 'numpy']
      )
