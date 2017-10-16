#include "stdlib.h"

#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *NoSuchFileError;
static PyObject *OutOfMemoryError;


char * load_file_into_buffer(char *name)
{
    FILE *file;
    if (!(file = fopen(name, "rb")))
    {
        PyErr_Format(NoSuchFileError, "Could not find file '%s'.", name);
        return NULL;
    }

    //Get file length
    fseek(file, 0, SEEK_END);
    unsigned long fileLen=ftell(file);
    fseek(file, 0, SEEK_SET);

    //Allocate memory
    char * buffer=(char *) malloc(fileLen+1);
    if (!buffer)
    {
        PyErr_SetString(OutOfMemoryError, "Not enough memory to load in file");
        return NULL;
    }

    //Read file contents into buffer
    fread(buffer, fileLen, 1, file);
    fclose(file);

    return buffer;
}

static PyObject *from_file(PyObject *dummy, PyObject *args)
{
    char *filepath=NULL;
    if (!PyArg_ParseTuple(args, "s", &filepath))
    {
        return NULL;
    }

    char *buffer = load_file_into_buffer(filepath);
    if(!buffer)
    {
        return  NULL;
    }

    puts(buffer);
    free(buffer);
    npy_intp dims[1] = {1};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if(!arr)
    {
        return NULL;
    }

    return arr;
}

static struct PyMethodDef methods[] = {
    {"from_file", from_file, METH_VARARGS, "descript of example"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initsssfile()
{
    PyObject *module = NULL;
    if(!(module = Py_InitModule("sssfile", methods)))
    {
        return;
    }

    PyObject *dict = NULL;
    if(!(dict = PyModule_GetDict(module)))
    {
        return;
    }

    NoSuchFileError = PyErr_NewException("sssfile.NoSuchFileError", NULL, NULL);
    PyDict_SetItemString(dict, "NoSuchFileError", NoSuchFileError);

    OutOfMemoryError = PyErr_NewException("sssfile.OutOfMemoryError", NULL, NULL);
    PyDict_SetItemString(dict, "OutOfMemoryError", OutOfMemoryError);

    import_array();
}
