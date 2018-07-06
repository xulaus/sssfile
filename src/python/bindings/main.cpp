#include "stdlib.h"
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#include "sssfile/column_builder.h"
#include "sssfile/metadata_reader.h"

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
    *buffer = (char *) malloc(fileLen + 1);
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

PyArrayObject *load_column_from_buffer(const char *buffer, int length,
                                       const SSSFile::sss_column_metadata &column_details)
{
    SSSFile::SSSError error = SSSFile::SUCCESS;

    unsigned int array_length = column_length_from_cstr(buffer, column_details);
    npy_intp dims[1] = {array_length};

    PyArray_Descr *dtype = NULL;

    switch (column_details.type)
    {
        case SSSFile::sss_column_metadata::TYPE_UTF32:
            dtype = PyArray_DescrNewFromType(NPY_UNICODE);
            dtype->elsize = column_details.size << 2;
            break;
        case SSSFile::sss_column_metadata::TYPE_UTF8:
            dtype = PyArray_DescrNewFromType(NPY_STRING);
            dtype->elsize = column_details.size;
            break;
        case SSSFile::sss_column_metadata::TYPE_DOUBLE:
            dtype = PyArray_DescrNewFromType(NPY_DOUBLE);
            break;
        case SSSFile::sss_column_metadata::TYPE_INT32:
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

    PyArrayObject *arr = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dims, dtype);

    if (!arr)
    {
        PyErr_NoMemory();
        return NULL;
    }

    if ((error = SSSFile::fill_column_from_substr(arr->data, buffer, length, column_details)) != SSSFile::SUCCESS)
    {
        Py_DECREF((PyObject *) arr);
        PyErr_Format(FailedToConvert, "Failed to convert file! %s", SSSFile::get_error_message(error));
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

    SSSFile::sss_column_metadata column_details = {};
    column_details.type = SSSFile::sss_column_metadata::TYPE_UTF32;
    column_details.line_length = get_line_length(buffer, buffer_length);
    column_details.size = column_details.line_length - 1;
    column_details.offset = 0;

    PyObject *arr = (PyObject *) load_column_from_buffer(buffer, buffer_length, column_details);

    free(buffer);
    return arr;
}

static PyObject *from_xmlfile(PyObject *dummy, PyObject *args)
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

    SSSFile::column_iterator *iter = NULL;
    SSSFile::xml_file_to_column_iterator(buffer, buffer_length, &iter);
    if (!iter)
    {
        printf("Failed to load XML file\n");
        return NULL;
    }

    do
    {
        SSSFile::sss_column_metadata column_details;
        if (find_column_details(iter, &column_details))
        {
            char buf[255];
            size_t length = find_column_name(iter, buf, 255);
            if (length > 0)
            {
                printf("'%.*s' is size %lu\n", (int) length, buf, column_details.size);
            }
            else
            {
                printf("Unnamed column is size %lu\n", column_details.size);
            }
        }
    } while (SSSFile::next_column(iter));

    SSSFile::free_column_iterator(iter);
    free(buffer);
    Py_INCREF(Py_None);
    return Py_None;
}

static struct PyMethodDef methods[] = {{"from_file", from_file, METH_VARARGS, ""},
                                       {"from_xmlfile", from_xmlfile, METH_VARARGS, ""},
                                       {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "sssfile", NULL, 0, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_sssfile()
{
    PyObject *module = NULL;
    if (!(module = PyModule_Create(&moduledef)))
    {
        return NULL;
    }

    PyObject *dict = NULL;
    if (!(dict = PyModule_GetDict(module)))
    {
        return NULL;
    }

    NoSuchFileError = PyErr_NewException("sssfile.NoSuchFileError", NULL, NULL);
    PyDict_SetItemString(dict, "NoSuchFileError", NoSuchFileError);

    FailedToConvert = PyErr_NewException("sssfile.FailedToConvert", NULL, NULL);
    PyDict_SetItemString(dict, "FailedToConvert", FailedToConvert);

    UnknownTypeError = PyErr_NewException("sssfile.UnknownTypeError", NULL, NULL);
    PyDict_SetItemString(dict, "UnknownTypeError", UnknownTypeError);

    import_array();

    return module;
}
