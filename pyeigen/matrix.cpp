#include "matrix.h"

static inline ModuleState* get_module_state(PyObject* module) {
    void* state = PyModule_GetState(module);
    assert(state != NULL);
    return (ModuleState*)state;
}
static struct PyModuleDef module;
// static inline ModuleState* get_module_state_by_def(PyTypeObject* tp)
// {
//     PyObject* mod = PyType_GetModuleByDef(tp, &module);
//     assert(mod != NULL);
//     return get_module_state(mod);
// }
static PyObject* MatrixObject_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    MatrixObject* self;
    self         = (MatrixObject*)type->tp_alloc(type, 0);
    self->shape  = Py_BuildValue("(ii)", 0, 0);
    self->device = Py_BuildValue("s", "cpu");
    // self->dtype  = PyArray_PyFloatDType;
    self->dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    self->ndim  = 0;
    self->size  = 0;

    return (PyObject*)self;
}
static void MatrixObject_dealloc(PyObject* self) {
    MatrixObject* obj = (MatrixObject*)self;
    delete obj->matrix;

    Py_XDECREF(obj->shape);
    Py_XDECREF(obj->device);
    Py_XDECREF(obj->dtype);

    Py_TYPE(self)->tp_free(self);
}
static int MatrixObject_init(PyObject* self, PyObject* args, PyObject* kwargs) {

    const char* kwlist[] = {"rows", "cols", NULL};
    int rows             = 0;
    int cols             = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", const_cast<char**>(kwlist), &rows,
                                     &cols)) {
        Py_DECREF(self);
        return NULL;
    }
    if (rows <= 0 or cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "The cols and rows must be greater than 0.");
        Py_DECREF(self);
        return NULL;
    }

    MatrixObject* obj = (MatrixObject*)self;

    obj->matrix = new Matrix(rows, cols);
    if (!obj->matrix) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix");
        Py_DECREF(self);
        return NULL;
    }
    Py_XSETREF(obj->shape, Py_BuildValue("(ii)", rows, cols));
    obj->ndim = 2;
    obj->size = rows * cols;

    return 0;
}

static PyObject* MatrixObject_repr(PyObject* self) {
    MatrixObject* obj = (MatrixObject*)self;
    Eigen::IOFormat format(Eigen::StreamPrecision, 0, ", ", ", ", "[", "]", "[", "]");
    std::stringstream ss;
    ss << obj->matrix->format(format);
    std::string matrix_str = ss.str();
    return PyUnicode_FromString(matrix_str.c_str());
    // return PyUnicode_FromFormat("<Matrix ndim=%d, shape=%S>", self->ndim,
    // self->shape);
}

static int MatrixObject_getbuffer(PyObject* self, Py_buffer* view, int flags) {
    MatrixObject* mat = (MatrixObject*)self;
    if (!mat->matrix) {
        PyErr_SetString(PyExc_ValueError, "Matrix is not initialized");
        return -1;
    }
    int rows         = mat->matrix->rows();
    int cols         = mat->matrix->cols();
    view->ndim       = 2;
    view->shape      = new Py_ssize_t[2]{rows, cols};
    view->strides    = new Py_ssize_t[2]{static_cast<Py_ssize_t>(cols * sizeof(double)),
                                         static_cast<Py_ssize_t>(sizeof(double))};
    view->itemsize   = sizeof(double);
    view->format     = const_cast<char*>("d");
    view->buf        = (void*)mat->matrix->data();
    view->readonly   = 1;
    view->suboffsets = nullptr;
    view->internal   = nullptr;
    return 0;
}
static void MatrixObject_releasebuffer(PyObject*, Py_buffer* view) {
    delete[] view->shape;
    delete[] view->strides;
}
inline Matrix* ParseMatrix(PyObject* obj) { return ((MatrixObject*)obj)->matrix; }
inline PyObject* ReturnMatrix(Matrix* m, PyTypeObject* type) {
    MatrixObject* obj = PyObject_NEW(MatrixObject, type);
    obj->matrix       = m;
    Py_XSETREF(obj->shape, Py_BuildValue("(ii)", m->rows(), m->cols()));
    Py_XSETREF(obj->device, Py_BuildValue("s", "cpu"));
    obj->dtype = PyArray_DescrFromType(NPY_DOUBLE);
    obj->ndim  = 2;
    obj->size  = m->rows() * m->cols();

    return (PyObject*)obj;
}

// matrix getset
PyObject* MatrixObject_data(PyObject* self, void* closure) {

    MatrixObject* obj = (MatrixObject*)self;
    Py_ssize_t rows   = obj->matrix->cols();
    Py_ssize_t cols   = obj->matrix->rows();

    PyObject* list = PyList_New(cols);
    for (int i = 0; i < cols; i++) {
        PyObject* internal = PyList_New(rows);

        for (int j = 0; j < rows; j++) {
            PyObject* value = PyFloat_FromDouble((*obj->matrix)(i, j));
            PyList_SetItem(internal, j, value);
        }

        PyList_SetItem(list, i, internal);
    }
    return list;
}

PyObject* MatrixObject_rows(PyObject* self, void* closure) {
    MatrixObject* obj = (MatrixObject*)self;
    return Py_BuildValue("i", obj->matrix->rows());
}

PyObject* MatrixObject_cols(PyObject* self, void* closure) {
    return Py_BuildValue("i", ((MatrixObject*)self)->matrix->cols());
}
PyObject* MatrixObject_ndim(PyObject* self, void* closure) {

    return Py_BuildValue("i", ((MatrixObject*)self)->ndim);
}
PyObject* MatrixObject_size(PyObject* self, void* closure) {

    return Py_BuildValue("i", ((MatrixObject*)self)->size);
}
PyObject* MatrixObject_shape(PyObject* self, void* closure) {
    return Py_XNewRef(((MatrixObject*)self)->shape);
}
PyObject* MatrixObject_device(PyObject* self, void* closure) {
    return Py_XNewRef(((MatrixObject*)self)->device);
}

PyObject* MatrixObject_dtype(PyObject* self, void* closure) {
    return Py_XNewRef(((MatrixObject*)self)->dtype);
}
static PyGetSetDef MatrixGetSet[] = {
    // { "data", (getter)MatrixObject_data, nullptr, nullptr },
    // { "rows", (getter)MatrixObject_rows, nullptr, nullptr },
    // { "cols", (getter)MatrixObject_cols, nullptr, nullptr },
    {"ndim", (getter)MatrixObject_ndim, nullptr, PyDoc_STR("Dim of matrix.")},
    {"size", (getter)MatrixObject_size, nullptr, PyDoc_STR("Size of matrix.")},
    {"shape", (getter)MatrixObject_shape, nullptr, PyDoc_STR("Shape of the matrix.")},
    {"device", (getter)MatrixObject_device, nullptr, PyDoc_STR("Device of the matrix.")},
    {"dtype", (getter)MatrixObject_dtype, nullptr, PyDoc_STR("Data type of the matrix.")},

    {nullptr}};

// number methods
static PyObject* Matrix_add(PyObject* a, PyObject* b) {
    Matrix* matrix_a = ParseMatrix(a);
    Matrix* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->cols() or matrix_a->rows() != matrix_b->rows()) {
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    Matrix* matrix_c = new Matrix(matrix_a->cols(), matrix_b->rows());
    *matrix_c        = *matrix_a + *matrix_b;

    return ReturnMatrix(matrix_c, a->ob_type);
}

static PyObject* Matrix_minus(PyObject* a, PyObject* b) {
    Matrix* matrix_a = ParseMatrix(a);
    Matrix* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->cols() or matrix_a->rows() != matrix_b->rows()) {
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    Matrix* matrix_c = new Matrix(matrix_a->cols(), matrix_b->rows());
    *matrix_c        = *matrix_a - *matrix_b;
    return ReturnMatrix(matrix_c, a->ob_type);
}

static PyObject* Matrix_multiply(PyObject* a, PyObject* b) {
    Matrix* matrix_a = ParseMatrix(a);
    Matrix* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->rows()) {
        PyErr_SetString(
            PyExc_ValueError,
            "The colonm rank of matrix A must be the same as the row rank of matrix B.");
        return NULL;
    }
    Matrix* matrix_c = new Matrix(matrix_a->rows(), matrix_b->cols());
    *matrix_c        = (*matrix_a) * (*matrix_b);
    return ReturnMatrix(matrix_c, a->ob_type);
}

// matrix methods
PyObject* MatrixObject_tolist(PyObject* self, PyObject* args) {
    return MatrixObject_data(self, nullptr);
}
static PyObject* MatrixObject_array(MatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    npy_intp dims[2] = {self->matrix->rows(), self->matrix->cols()};

    PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, self->matrix->data());
    if (!arr)
        return nullptr;

    PyArray_SetBaseObject((PyArrayObject*)arr, (PyObject*)self);
    Py_INCREF(self);

    return arr;
}

static PyMethodDef matrix_methods[] = {
    {"to_list", (PyCFunction)MatrixObject_tolist, METH_VARARGS,
     PyDoc_STR("Return the matrix data to a list object.")},
    {"__array__", (PyCFunction)MatrixObject_array, METH_NOARGS,
     PyDoc_STR("Return the matrix data to a numpy array object.")},
    {"numpy", (PyCFunction)MatrixObject_array, METH_NOARGS,
     PyDoc_STR("Return the matrix data to a numpy array object.")},
    {"to_numpy", (PyCFunction)MatrixObject_array, METH_NOARGS,
     PyDoc_STR("Return the matrix data to a numpy array object.")},
    {nullptr, nullptr}

};

static PyType_Slot matrix_slots[] = {{Py_tp_new, (void*)(newfunc)MatrixObject_new},
                                     {Py_tp_init, (void*)(initproc)MatrixObject_init},
                                     {Py_tp_dealloc, (void*)(destructor)MatrixObject_dealloc},
                                     {Py_tp_repr, (void*)(reprfunc)MatrixObject_repr},
                                     {Py_tp_methods, matrix_methods},
                                     {Py_bf_getbuffer, (void*)MatrixObject_getbuffer},
                                     {Py_bf_releasebuffer, (void*)MatrixObject_releasebuffer},
                                     // { Py_tp_members, Complex_members },
                                     {Py_tp_getset, MatrixGetSet},
                                     // Number protocol
                                     {Py_nb_add, (void*)(binaryfunc)Matrix_add},
                                     {Py_nb_subtract, (void*)(binaryfunc)Matrix_minus},
                                     {Py_nb_multiply, (void*)(binaryfunc)Matrix_multiply},
                                     {Py_nb_matrix_multiply, (void*)(binaryfunc)Matrix_multiply},
                                     {0, nullptr}};
static PyType_Spec matrix_spec    = {.name      = "matrix",
                                     .basicsize = sizeof(MatrixObject),
                                     .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                     .slots     = matrix_slots};

// module methods
static PyObject* Matrix_ones(PyObject* self, PyObject* args, PyObject* kwargs) {
    ModuleState* state = get_module_state(self);
    MatrixObject* m    = (MatrixObject*)MatrixObject_new(state->matrix_type, args, kwargs);
    MatrixObject_init((PyObject*)m, args, kwargs);
    m->matrix->setOnes();
    return (PyObject*)m;
}

static PyObject* Matrix_zeros(PyObject* self, PyObject* args, PyObject* kwargs) {
    ModuleState* state = get_module_state(self);
    MatrixObject* m    = (MatrixObject*)MatrixObject_new(state->matrix_type, args, kwargs);
    MatrixObject_init((PyObject*)m, args, kwargs);
    m->matrix->setZero();
    return (PyObject*)m;
}

static PyObject* Matrix_random(PyObject* self, PyObject* args, PyObject* kwargs) {
    // PyTypeObject* MatrixType = (PyTypeObject*)PyType_FromSpec(&matrix_spec);
    ModuleState* state = get_module_state(self);
    MatrixObject* m    = (MatrixObject*)MatrixObject_new(state->matrix_type, args, kwargs);
    MatrixObject_init((PyObject*)m, args, kwargs);
    m->matrix->setRandom();
    return (PyObject*)m;
}

static PyObject* Matrix_from_list(PyObject* self, PyObject* args) {
    PyObject* data = nullptr;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object.");
        return nullptr;
    }
    if (!PyList_Check(data)) {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object.");
        return nullptr;
    }
    int cols = PyList_GET_SIZE(data);
    if (cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object.");
        return nullptr;
    }
    PyObject* list = PyList_GET_ITEM(data, 0);
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object.");
        return nullptr;
    }
    int rows      = PyList_GET_SIZE(list);
    Matrix* p_mat = new Matrix(rows, cols);
    for (int i = 0; i < cols; i++) {
        PyObject* list = PyList_GET_ITEM(data, i);
        if (!PyList_Check(list)) {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object.");
            return nullptr;
        }
        int tmp = PyList_GET_SIZE(list);
        if (rows != tmp) {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. Each "
                                              "elements of it must be the same length.");
            return nullptr;
        }
        rows = tmp;

        for (int j = 0; j < rows; j++) {
            PyObject* num = PyList_GET_ITEM(list, j);
            if (!PyFloat_Check(num)) {
                PyErr_SetString(PyExc_ValueError, "Every elements of the matrix must float.");
                return nullptr;
            }
            (*p_mat)(i, j) = ((PyFloatObject*)num)->ob_fval;
        }
    }
    ModuleState* state = get_module_state(self);
    return ReturnMatrix(p_mat, state->matrix_type);
}

static PyObject* Matrix_from_numpy(PyObject* self, PyObject* args) {
    // PyObject* np_array;
    // if (!PyArg_ParseTuple(args, "O", &np_array))
    // {
    //     return nullptr;
    // }
    if (!PyArray_Check(args)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
        return nullptr;
    }
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(args, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!array) {
        return nullptr;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional");
        Py_DECREF(array);
        return nullptr;
    }
    int rows     = PyArray_DIM(array, 0);
    int cols     = PyArray_DIM(array, 1);
    double* data = (double*)malloc(rows * cols * sizeof(double));
    if (!data) {
        Py_DECREF(array);
        Py_DECREF(self);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix");
        return NULL;
    }
    double* np_data = (double*)PyArray_DATA(array);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = np_data[i];
    }

    Py_DECREF(array);

    Matrix* matrix = new Matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            (*matrix)(i, j) = *((MATRIX_DEFAULT_TYPE*)PyArray_GETPTR2(array, i, j));
        }
    }
    ModuleState* state = get_module_state(self);
    return ReturnMatrix(matrix, state->matrix_type);
}

static PyMethodDef module_methods[] = {
    {"ones", (PyCFunction)Matrix_ones, METH_VARARGS | METH_KEYWORDS,
     "Return a new matrix with initial values one."},
    {"zeros", (PyCFunction)Matrix_zeros, METH_VARARGS | METH_KEYWORDS,
     "Return a new matrix with initial values zero."},
    {"random", (PyCFunction)Matrix_random, METH_VARARGS | METH_KEYWORDS,
     "Return a new matrix with random values"},
    {"from_list", (PyCFunction)Matrix_from_list, METH_VARARGS,
     "Return a new matrix with given values"},
    {"from_numpy", (PyCFunction)Matrix_from_numpy, METH_O,
     "Return a new matrix with given numpy array"},
    {NULL, NULL, 0, NULL}};

static int module_traverse(PyObject* module, visitproc visit, void* arg) {
    ModuleState* state = get_module_state(module);
    Py_VISIT(state->matrix_type);
    return 0;
}
static int module_clear(PyObject* module) {
    ModuleState* state = get_module_state(module);
    Py_CLEAR(state->matrix_type);
    return 0;
}

static void module_free(void* module) { module_clear((PyObject*)module); }
/* Initialize this module. */
static int module_exec(PyObject* module) {
    ModuleState* state = get_module_state(module);
    state->matrix_type = (PyTypeObject*)PyType_FromModuleAndSpec(module, &matrix_spec, nullptr);
    if (PyModule_AddObjectRef(module, "matrix", (PyObject*)state->matrix_type) < 0) {
        return -1;
    }
    // heap type
    // PyObject* matrix_type = PyType_FromSpec(&matrix_spec);
    // if (matrix_type == NULL)
    // {
    //     return -1;
    // }
    // if (PyModule_AddObjectRef(module, "Matrix", matrix_type) < 0)
    // {
    //     Py_DECREF(matrix_type);  //Failed to register, release the class
    //     object return -1;
    // }
    // Py_DECREF(matrix_type);
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    import_array();
    return 0;
}

static PyModuleDef_Slot module_slots[] = {{Py_mod_exec, (void*)module_exec},
#if PY_VERSION_HEX >= 0x030D0000
                                          {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
                                          {0, NULL}};
PyDoc_STRVAR(module_doc,
             "Implementation module for SSL socket operations.  See the socket module\n\
for documentation.");

// static struct PyModuleDef module = {
//     .m_base = PyModuleDef_HEAD_INIT, .m_name = "matrix", .m_doc = module_doc,
//     .m_size = sizeof(ModuleState), .m_methods = module_methods, .m_slots =
//     module_slots, .m_traverse = module_traverse, .m_clear = module_clear,
//     .m_free = module_free
// };

PyMODINIT_FUNC PyInit_matrix(void) {
    module.m_base     = PyModuleDef_HEAD_INIT;
    module.m_name     = "matrix";
    module.m_doc      = module_doc;
    module.m_size     = sizeof(ModuleState);
    module.m_methods  = module_methods;
    module.m_slots    = module_slots;
    module.m_traverse = module_traverse;
    module.m_clear    = module_clear;
    module.m_free     = module_free;
    return PyModuleDef_Init(&module);
}
