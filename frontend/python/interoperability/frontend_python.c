#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <dlfcn.h>

#include "backend_native.h"

// TODO: document methods

static PyObject* frontend_native_forward(PyObject* self, PyObject* args) {
    const PyBytesObject* b;
    int s, rs;

    if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
        return NULL;

    float* m = (float*) PyBytes_AsString((PyObject*) b);
    float ret[rs];

    void* native_backend_handle = dlopen("libbackend_native.so", RTLD_LAZY);
    if (!native_backend_handle) {
        PyErr_SetString(PyExc_IOError, "Could not open native backend library");
        return NULL;
    }
    dlerror();

    NATIVE_FORWARD_T* native_forward = dlsym(native_backend_handle, "native_forward");
    if (dlerror()) {
        PyErr_SetString(PyExc_IOError, "Could not find native_forward in library");
        return NULL;
    }

    int sts = (*native_forward)(m, s, ret, rs);
    if (sts) {
        PyErr_SetString(PyExc_IOError, "Error during native forward");
        return NULL; // TODO: do some error handling
    }

    dlclose(native_backend_handle);

    return Py_BuildValue("y#", ret, rs * sizeof(float));
}

#ifdef SGX_SDK
static PyObject *frontend_enclave_forward(PyObject *self, PyObject *args) {
  const PyBytesObject *b;
  int s,rs;

  if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
    return NULL;

  float *m = (float *)PyBytes_AsString((PyObject *)b);
  float ret[rs];
  // printf("Enclave NN forward\n");

  int sts = enclave_nn_forward(m, s, ret, rs);
  if (sts){
    PyErr_SetString(PyExc_IOError, "Error in sgx");
    return NULL; // TODO: do some error handling
  }


  return Py_BuildValue("y#", ret, rs*sizeof(float));
}

static PyObject *enclave_initialize(PyObject *self, PyObject *args){
  enclave_nn_start();
  return Py_None;
}

static PyObject *enclave_teardown(PyObject *self, PyObject *args){
  enclave_nn_end();
  return Py_None;
}
#endif

static PyMethodDef frontend_methods[] = {
#ifdef SGX_SDK
        {"initialize", enclave_initialize, METH_VARARGS, "Initialize matutil"},
        {"teardown", enclave_teardown, METH_VARARGS, "Teardown matutil"},
        {"enclave_forward", frontend_enclave_forward, METH_VARARGS, "Execute forward pass of all layers moved to TEE"},
#endif
        {"native_forward", frontend_native_forward, METH_VARARGS, "Execute forward pass of sgx layers in native C"},
        {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef frontend_python = {
        PyModuleDef_HEAD_INIT,
        "frontend_python", // module name
        NULL,        // module documentation, may be NULL
        -1, // size of per-interpreter state of the module,or -1 if the module
        // keeps state in global variables.
        frontend_methods};

PyMODINIT_FUNC // macro declares return type, special linkage, and "extern C"
// if needed
PyInit_frontend_python(void) {
    return PyModule_Create(&frontend_python);
}
