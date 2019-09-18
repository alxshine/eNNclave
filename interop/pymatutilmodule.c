#define PY_SSIZE_T_CLEAN
#include <python3.7m/Python.h>

#include "matutil.h"

static PyObject *pymatutil_dump_matrix(PyObject *self, PyObject *args) {
  const PyBytesObject *bytes;
  int cols;
  int rows;
  char *data;
  Py_ssize_t data_length;

  if (!PyArg_ParseTuple(args, "Sii", &bytes, &rows, &cols))
    return NULL;

  if (PyBytes_AsStringAndSize((PyObject *)bytes, &data, &data_length))
    return NULL;

  float *array = (float *)data;
  matutil_dump_matrix(array, cols, rows); // call into libmatutil.so

  Py_RETURN_NONE;
}

static PyObject *pymatutil_test_bytes(PyObject *self, PyObject *args) {
  // return range(10) to test return value interoperability
  int test[10];
  for (int i = 0; i < 10; ++i) {
    test[i] = i;
  }

  PyObject *ret = PyBytes_FromStringAndSize((char *)test, 40);
  return ret;
}

static PyObject *pymatutil_multiply(PyObject *self, PyObject *args) {
  const PyBytesObject *b1, *b2;
  int w1, h1, w2, h2;

  if (!PyArg_ParseTuple(args, "SiiSii", &b1, &w1, &h1, &b2, &w2, &h2))
    return NULL;

  float *m1, *m2;
  m1 = (float *)PyBytes_AsString((PyObject *)b1);
  if(!m1)
    return NULL;
  m2 = (float *)PyBytes_AsString((PyObject *)b2);
  if(!m2)
    return NULL;
  
  int wr, hr;
  matutil_get_new_dimensions(w1, h1, w2, h2, &wr, &hr);
  float mret[wr*hr];
  int status = matutil_multiply(m1, w1, h1, m2, w2, h2, mret);
  if(status)
    return NULL; //TODO: do some meaningful error handling/exception raising

  return PyBytes_FromStringAndSize((char *)mret, wr*hr*sizeof(float));
}

static PyObject *pymatutil_add(PyObject *self, PyObject *args){
  const PyBytesObject *b1, *b2;
  int w1, h1, w2, h2;

  if (!PyArg_ParseTuple(args, "SiiSii", &b1, &w1, &h1, &b2, &w2, &h2))
    return NULL;

  float *m1, *m2;
  m1 = (float *)PyBytes_AsString((PyObject *)b1);
  if(!m1)
    return NULL;
  m2 = (float *)PyBytes_AsString((PyObject *)b2);
  if(!m2)
    return NULL;
  
  int wr, hr;
  matutil_get_new_dimensions(w1, h1, w2, h2, &wr, &hr);
  float mret[wr*hr];
  int status = matutil_add(m1, w1, h1, m2, w2, h2, mret);
  if(status)
    return NULL; //TODO: do some meaningful error handling/exception raising

  return PyBytes_FromStringAndSize((char *)mret, wr*hr*sizeof(float));
}

static PyObject *pymatutil_relu(PyObject *self, PyObject *args){
  const PyBytesObject *b;
  int w,h;

  if (!PyArg_ParseTuple(args, "Sii", &b, &w, &h))
    return NULL;

  float *m = (float*) PyBytes_AsString((PyObject*) b);
  if(!m)
    return NULL; //TODO: raise exception

  float mret[w*h];
  memcpy(mret, m, w*h*sizeof(float));
  matutil_relu(mret, w, h);
  return PyBytes_FromStringAndSize((char *)mret, w*h*sizeof(float));
}

static PyMethodDef PymatutilMethods[] = {
    {"dump_matrix", pymatutil_dump_matrix, METH_VARARGS, "Dump matrix"},
    {"test_bytes", pymatutil_test_bytes, METH_VARARGS, "Test bytes"},
    {"multiply", pymatutil_multiply, METH_VARARGS, "Multiply matrices"},
    {"add", pymatutil_add, METH_VARARGS, "Add matrices"},
    {"relu", pymatutil_relu, METH_VARARGS, "Execute ReLU on matrix"},
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