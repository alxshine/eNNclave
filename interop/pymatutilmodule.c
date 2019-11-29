#define PY_SSIZE_T_CLEAN
#ifdef HOST
#include <python3.7m/Python.h>
#else
#include <python3.6m/Python.h>
#endif

#include "matutil.hpp"

static PyObject *pymatutil_test_bytes(PyObject *self, PyObject *args) {
  // return range(10) to test return value interoperability
  int test[10];
  for (int i = 0; i < 10; ++i) {
    test[i] = i;
  }

  PyObject *ret = PyBytes_FromStringAndSize((char *)test, 40);
  return ret;
}

static PyObject *pymatutil_forward(PyObject *self, PyObject *args) {
  const PyBytesObject *b;
  int r, c;

  if (!PyArg_ParseTuple(args, "Sii", &b, &r, &c))
    return NULL;

  float *m = (float *)PyBytes_AsString((PyObject *)b);
  int label;
  int sts = matutil_forward(m, r*c, &label);
  if (sts){
    PyErr_SetString(PyExc_IOError, "Error in enclave");
    return NULL; // TODO: do some error handling
  }

  return PyLong_FromLong(label);
}

static PyObject *pymatutil_initialize(PyObject *self, PyObject *args){
  matutil_initialize();
  return Py_None;
}

static PyObject *pymatutil_teardown(PyObject *self, PyObject *args){
  matutil_teardown();
  return Py_None;
}

static PyMethodDef PymatutilMethods[] = {
    {"initialize", pymatutil_initialize, METH_VARARGS, "Initialize matutil"},
    {"teardown", pymatutil_teardown, METH_VARARGS, "Teardown matutil"},
    {"forward", pymatutil_forward, METH_VARARGS, "Execute forward pass of all layers moved to TEE"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef pymatutilmodule = {
    PyModuleDef_HEAD_INIT,
    "pymatutil", // module name
    NULL,        // module documentation, may be NULL
    -1, // size of per-interpreter state of the module,or -1 if the module
        // keeps state in global variables.
    PymatutilMethods};

PyMODINIT_FUNC // macro declares return type, special linkage, and "extern C"
               // if needed
PyInit_pymatutil(void) {
  return PyModule_Create(&pymatutilmodule);
}
