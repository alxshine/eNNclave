#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <dlfcn.h>
#include <stdlib.h>

#include "backends.h"

// TODO: document methods

static void* load_library(const char* library_name) {
    char library_path[256];
    char* ennclave_home = getenv("ENNCLAVE_HOME");
    if (ennclave_home == NULL) {
        fprintf(stderr, "Environment variable ENNCLAVE_HOME not set, defaulting to current working directory\n");
        ennclave_home = ".";
    }

    int num_written = snprintf(library_path, sizeof(library_path), "%s/lib/libbackend_%s.so", ennclave_home,
                               library_name);
    if(num_written >= sizeof(library_path)){
        perror("Library path too long for buffer");
        exit(1);
    }
    printf("Loading library %s\n", library_path);
    return dlopen(library_path, RTLD_NOW);
}

static PyObject* frontend_native_forward(PyObject* self, PyObject* args) {
    const PyBytesObject* b;
    int s, rs;

    if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
        return NULL;

    float* m = (float*) PyBytes_AsString((PyObject*) b);
    float ret[rs];

    void* native_backend_handle = load_library("native");
    if (!native_backend_handle) {
        PyErr_SetString(PyExc_IOError, "Could not open native backend library");
        return NULL;
    }
    dlerror();

    FORWARD_T* native_forward = dlsym(native_backend_handle, "native_forward");
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
static PyObject *frontend_sgx_forward(PyObject *self, PyObject *args) {
  const PyBytesObject *b;
  int s,rs;

  if (!PyArg_ParseTuple(args, "Sii", &b, &s, &rs))
    return NULL;

  float *m = (float *)PyBytes_AsString((PyObject *)b);
  float ret[rs];
  printf("Enclave NN forward\n");

  void* sgx_backend_handle = load_library("sgx", RTLD_NOW);
    if (!sgx_backend_handle) {
        PyErr_SetString(PyExc_IOError, "Could not open native backend library");
        return NULL;
    }
    dlerror();

    FORWARD_T* sgx_forward = dlsym(sgx_backend_handle, "sgx_forward");
    if (dlerror()) {
        PyErr_SetString(PyExc_IOError, "Could not find sgx_forward in library");
        return NULL;
    }

    int sts = (*sgx_forward)(m, s, ret, rs);
    if (sts) {
        PyErr_SetString(PyExc_IOError, "Error during SGX forward");
        return NULL; // TODO: do some error handling
    }

  return Py_BuildValue("y#", ret, rs*sizeof(float));
}

static PyObject *enclave_initialize(PyObject *self, PyObject *args){
  // enclave_nn_start();
  return Py_None;
}

static PyObject *enclave_teardown(PyObject *self, PyObject *args){
  // enclave_nn_end();
  return Py_None;
}
#endif

static PyMethodDef frontend_methods[] = {
#ifdef SGX_SDK
        {"sgx_forward", frontend_sgx_forward, METH_VARARGS, "Execute forward pass of all layers moved to TEE"},
#endif
        {"native_forward", frontend_native_forward, METH_VARARGS, "Execute forward pass of enclave layers in native C"},
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
