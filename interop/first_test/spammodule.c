#define PY_SSIZE_T_CLEAN
#include <python3.7m/Python.h>

static PyObject *spam_print(PyObject *self, PyObject *args) {
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  printf("Being printed from C:\n");
  printf("%s\n", command);

  Py_RETURN_NONE;
}

static PyObject *spam_print_array(PyObject *self, PyObject *args) {
  const PyBytesObject *bytes;
  int cols;
  int rows;
  char *data;
  Py_ssize_t data_length;

  if (!PyArg_ParseTuple(args, "Sii", &bytes, &rows, &cols))
    return NULL;

  printf("Got dimensions: %dx%d\n", cols, rows);

  if(PyBytes_AsStringAndSize((PyObject *)bytes, &data, &data_length))
    return NULL;
  printf("data_length: %d\n", (int)data_length);

  /* for (Py_ssize_t i = 0; i < data_length; ++i) */
    /* printf("%02x ", data[i]); */
  /* printf("\n"); */

  float *array = (float*) data;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%f ", array[i*cols+j]);
    }
    printf("\n");
  }
  
  Py_RETURN_NONE;
}

static PyMethodDef SpamMethods[] = {
    {"print", spam_print, METH_VARARGS, "Print the argument"},
    {"print_array", spam_print_array, METH_VARARGS, "Print array contents"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam", // module name
    NULL,   // module documentation, may be NULL
    -1,     // size of per-interpreter state of the module,or -1 if the module
            // keeps state in global variables.
    SpamMethods};

PyMODINIT_FUNC // macro declares return type, special linkage, and "extern C"
               // if needed
PyInit_spam(void) {
  return PyModule_Create(&spammodule);
}
