#include "stdlib.h"

#include "Python.h"
#include "numpy/arrayobject.h"

#include "sssfile/column_builder.h"

static PyObject *NoSuchFileError;
static PyObject *FailedToConvert;
static PyObject *UnknownTypeError;

int load_file_into_buffer(char *name, char **buffer)
{
    FILE *file;
    if (!(file = fopen(name, "rb")))
    {
        PyErr_Format(NoSuchFileError, "Could not find file '%s'.", name);
        return -1;
    }

    // Get file length
    fseek(file, 0, SEEK_END);
    unsigned long fileLen = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory
    *buffer = (char *)malloc(fileLen + 1);
    if (!*buffer)
    {
        PyErr_NoMemory();
        return -2;
    }
    // Read file contents into buffer
    fread(*buffer, fileLen, 1, file);
    (*buffer)[fileLen] = '\0';
    fclose(file);

    return fileLen;
}

int get_line_length(char *buffer, int buffer_length)
{

    int line_length = 0;
    for (; line_length < buffer_length && buffer[line_length] != '\n'; line_length++)
    {
    }
    return ++line_length;
}

PyArrayObject *load_column_from_buffer(const char *buffer, const SSSFile::column_metadata &column_details)
{
    unsigned int array_length = column_length(buffer, column_details);
    npy_intp dims[1] = {array_length};

    PyArray_Descr *dtype = NULL;

    switch (column_details.type)
    {
    case SSSFile::column_metadata::TYPE_UTF32:
        dtype = PyArray_DescrNewFromType(NPY_UNICODE);
        dtype->elsize = column_details.size << 2;
        break;
    case SSSFile::column_metadata::TYPE_UTF8:
        dtype = PyArray_DescrNewFromType(NPY_STRING);
        dtype->elsize = column_details.size;
        break;
    case SSSFile::column_metadata::TYPE_DOUBLE:
        dtype = PyArray_DescrNewFromType(NPY_DOUBLE);
        break;
    case SSSFile::column_metadata::TYPE_INT32:
        dtype = PyArray_DescrNewFromType(NPY_INT32);
        break;
    default:
        break;
    }

    if (!dtype)
    {
        PyErr_SetString(UnknownTypeError, "Unknown Type");
        return NULL;
    }

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, dims, dtype);

    if (!arr)
    {
        PyErr_NoMemory();
        return NULL;
    }

    if (!SSSFile::fill_column(arr->data, buffer, column_details))
    {
        Py_DECREF((PyObject *)arr);
        PyErr_SetString(NoSuchFileError, "Failed to convert file");
        return NULL;
    }

    return arr;
}

static PyObject *from_file(PyObject *dummy, PyObject *args)
{
    char *filepath = NULL;
    if (!PyArg_ParseTuple(args, "s", &filepath))
    {
        return NULL;
    }

    char *buffer = NULL;
    int buffer_length = load_file_into_buffer(filepath, &buffer);
    if (!buffer || buffer_length < 0)
    {
        return NULL;
    }

    // Temp hack while there is no way to read colspecs
    const char *filepath2 = "tests/data/sss-2.0.dat";
    if (strcmp(filepath, filepath2) != 0)
    {
        PyErr_Format(NoSuchFileError, "Could not find file '%s'.", filepath);
        return NULL;
    }

    SSSFile::column_metadata column_details = {};
    column_details.type = SSSFile::column_metadata::TYPE_UTF8;
    column_details.line_length = get_line_length(buffer, buffer_length);
    column_details.size = column_details.line_length - 1;
    column_details.offset = 0;

    PyObject *arr = (PyObject *)load_column_from_buffer(buffer, column_details);

    free(buffer);
    return arr;
}

static struct PyMethodDef methods[] = {{"from_file", from_file, METH_VARARGS, "descript of example"},
                                       {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initsssfile()
{
    PyObject *module = NULL;
    if (!(module = Py_InitModule("sssfile", methods)))
    {
        return;
    }

    PyObject *dict = NULL;
    if (!(dict = PyModule_GetDict(module)))
    {
        return;
    }

    NoSuchFileError = PyErr_NewException("sssfile.NoSuchFileError", NULL, NULL);
    PyDict_SetItemString(dict, "NoSuchFileError", NoSuchFileError);

    FailedToConvert = PyErr_NewException("sssfile.FailedToConvert", NULL, NULL);
    PyDict_SetItemString(dict, "FailedToConvert", FailedToConvert);

    UnknownTypeError = PyErr_NewException("sssfile.UnknownTypeError", NULL, NULL);
    PyDict_SetItemString(dict, "UnknownTypeError", UnknownTypeError);

    import_array();
}
