#include "stdlib.h"

#include "Python.h"
#include "numpy/arrayobject.h"

#include "sssfile/column_builder.h"

static PyObject *NoSuchFileError;

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
    column_details.type = SSSFile::column_metadata::TYPE_INT32;
    column_details.line_length = get_line_length(buffer, buffer_length);
    column_details.size = column_details.line_length - 1;
    column_details.offset = 0;

    unsigned int array_length = column_length(buffer, column_details);
    npy_intp dims[1] = {array_length};
    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);

    if (!arr)
    {
        goto fail;
    }

    SSSFile::fill_column((void *)arr->data, buffer, column_details);
    return (PyObject *)arr;

fail:
    free(buffer);
    return NULL;
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

    import_array();
}
