#define PY_SSIZE_T_CLEAN
#include <python3.7m/Python.h>

#include "native_nn.h"
#include "enclave_nn.h"

static PyObject *pymatutil_native_forward(PyObject *self, PyObject *args) {
  const PyBytesObject *b;
  int s, rs;

  if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
    return NULL;

  float *m = (float *)PyBytes_AsString((PyObject *)b);
  float ret[rs];
  printf("Native NN forward\n");
  
  int sts = native_nn_forward(m, s, ret, rs);
  if (sts){
    PyErr_SetString(PyExc_IOError, "Error during native forward");
    return NULL; // TODO: do some error handling
  }

  return Py_BuildValue("y#", ret, rs*sizeof(float));
}

static PyObject *pymatutil_enclave_forward(PyObject *self, PyObject *args) {
  const PyBytesObject *b;
  int s,rs;

  if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
    return NULL;

  float *m = (float *)PyBytes_AsString((PyObject *)b);
  float ret[rs];
  printf("Enclave NN forward\n");

  int sts = enclave_nn_forward(m, s, ret, rs);
  if (sts){
    PyErr_SetString(PyExc_IOError, "Error in enclave");
    return NULL; // TODO: do some error handling
  }


  return Py_BuildValue("y#", ret, rs*sizeof(float));
}

static PyObject *pymatutil_forward(PyObject *self, PyObject *args){
  return pymatutil_enclave_forward(self, args);
}

static PyObject *enclave_initialize(PyObject *self, PyObject *args){
  enclave_nn_start();
  return Py_None;
}

static PyObject *enclave_teardown(PyObject *self, PyObject *args){
  enclave_nn_end();
  return Py_None;
}

static PyMethodDef PymatutilMethods[] = {
    {"initialize", enclave_initialize, METH_VARARGS, "Initialize matutil"},
    {"teardown", enclave_teardown, METH_VARARGS, "Teardown matutil"},
    {"enclave_forward", pymatutil_enclave_forward, METH_VARARGS, "Execute forward pass of all layers moved to TEE"},
    {"native_forward", pymatutil_native_forward, METH_VARARGS, "Execute forward pass of enclave layers in native C"},
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
