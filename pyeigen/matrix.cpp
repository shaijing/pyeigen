#ifdef USE_MKL
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif
#include <Eigen/Dense>
#include <Python.h>

struct Matrix
{
    PyObject_HEAD Eigen::MatrixXd* matrix = nullptr;
};
struct ModuleState
{
    PyTypeObject* matrix_type;
};

static inline ModuleState* get_module_state(PyObject* module)
{
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
static PyObject* Matrix_new(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    Matrix* self;
    self = (Matrix*)type->tp_alloc(type, 0);

    return (PyObject*)self;
}
static void Matrix_dealloc(PyObject* self)
{
    delete ((Matrix*)self)->matrix;
    Py_TYPE(self)->tp_free(self);
}
static int Matrix_init(PyObject* self, PyObject* args, PyObject* kwargs)
{

    const char* kwlist[] = { "rows", "cols", NULL };
    int rows             = 0;
    int cols             = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist,
                                     &rows, &cols))
    {
        Py_DECREF(self);
        return NULL;
    }
    if (rows <= 0 or cols <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "The cols and rows must be greater than 0.");
        return NULL;
    }

    ((Matrix*)self)->matrix = new Eigen::MatrixXd(rows, cols);
    return 0;
}

inline Eigen::MatrixXd* ParseMatrix(PyObject* obj)
{
    return ((Matrix*)obj)->matrix;
}
inline PyObject* ReturnMatrix(Eigen::MatrixXd* m, PyTypeObject* type)
{
    Matrix* obj = PyObject_NEW(Matrix, type);
    obj->matrix = m;
    return (PyObject*)obj;
}

// matrix getset
PyObject* Matrix_data(PyObject* self, void* closure)
{

    Matrix* obj     = (Matrix*)self;
    Py_ssize_t rows = obj->matrix->cols();
    Py_ssize_t cols = obj->matrix->rows();

    PyObject* list = PyList_New(cols);
    for (int i = 0; i < cols; i++)
    {
        PyObject* internal = PyList_New(rows);

        for (int j = 0; j < rows; j++)
        {
            PyObject* value = PyFloat_FromDouble((*obj->matrix)(i, j));
            PyList_SetItem(internal, j, value);
        }

        PyList_SetItem(list, i, internal);
    }
    return list;
}

PyObject* Matrix_rows(PyObject* self, void* closure)
{
    Matrix* obj = (Matrix*)self;
    return Py_BuildValue("i", obj->matrix->rows());
}

PyObject* Matrix_cols(PyObject* self, void* closure)
{
    Matrix* obj = (Matrix*)self;
    return Py_BuildValue("i", obj->matrix->cols());
}

static PyGetSetDef MatrixGetSet[] = {
    { "data", (getter)Matrix_data, nullptr, nullptr },
    { "rows", (getter)Matrix_rows, nullptr, nullptr },
    { "cols", (getter)Matrix_cols, nullptr, nullptr },
    { nullptr }
};

// number methods
static PyObject* Matrix_add(PyObject* a, PyObject* b)
{
    Eigen::MatrixXd* matrix_a = ParseMatrix(a);
    Eigen::MatrixXd* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->cols() or matrix_a->rows() != matrix_b->rows())
    {
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    Eigen::MatrixXd* matrix_c = new Eigen::MatrixXd(matrix_a->cols(), matrix_b->rows());
    *matrix_c                 = *matrix_a + *matrix_b;

    return ReturnMatrix(matrix_c, a->ob_type);
}

static PyObject* Matrix_minus(PyObject* a, PyObject* b)
{
    Eigen::MatrixXd* matrix_a = ParseMatrix(a);
    Eigen::MatrixXd* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->cols() or matrix_a->rows() != matrix_b->rows())
    {
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    Eigen::MatrixXd* matrix_c = new Eigen::MatrixXd(matrix_a->cols(), matrix_b->rows());
    *matrix_c                 = *matrix_a - *matrix_b;
    return ReturnMatrix(matrix_c, a->ob_type);
}

static PyObject* Matrix_multiply(PyObject* a, PyObject* b)
{
    Eigen::MatrixXd* matrix_a = ParseMatrix(a);
    Eigen::MatrixXd* matrix_b = ParseMatrix(b);

    if (matrix_a->cols() != matrix_b->rows())
    {
        PyErr_SetString(PyExc_ValueError, "The colonm rank of matrix A must be the same as the row rank of matrix B.");
        return NULL;
    }
    Eigen::MatrixXd* matrix_c = new Eigen::MatrixXd(matrix_a->rows(), matrix_b->cols());
    *matrix_c                 = (*matrix_a) * (*matrix_b);
    return ReturnMatrix(matrix_c, a->ob_type);
}

// matrix methods
PyObject* Matrix_tolist(PyObject* self, PyObject* args)
{
    return Matrix_data(self, nullptr);
}
static PyMethodDef matrix_methods[] = {
    { "to_list", (PyCFunction)Matrix_tolist, METH_VARARGS, PyDoc_STR("Return the matrix data to a list object.") },
    { nullptr, nullptr }

};

static PyType_Slot matrix_slots[] = {
    { Py_tp_new, (void*)(newfunc)Matrix_new },
    { Py_tp_init, (void*)(initproc)Matrix_init },
    { Py_tp_dealloc, (void*)(destructor)Matrix_dealloc },
    { Py_tp_methods, matrix_methods },
    // { Py_tp_members, Complex_members },
    { Py_tp_getset, MatrixGetSet },
    // Number protocol
    { Py_nb_add, (void*)(binaryfunc)Matrix_add },
    { Py_nb_subtract, (void*)(binaryfunc)Matrix_minus },
    { Py_nb_multiply, (void*)(binaryfunc)Matrix_multiply },
    { 0, nullptr }
};
static PyType_Spec matrix_spec = {
    .name      = "matrix",
    .basicsize = sizeof(Matrix),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots     = matrix_slots
};

// module methods
static PyObject* Matrix_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
    ModuleState* state = get_module_state(self);
    Matrix* m          = (Matrix*)Matrix_new(state->matrix_type, args, kwargs);
    Matrix_init((PyObject*)m, args, kwargs);
    m->matrix->setOnes();
    return (PyObject*)m;
}

static PyObject* Matrix_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
{
    ModuleState* state = get_module_state(self);
    Matrix* m          = (Matrix*)Matrix_new(state->matrix_type, args, kwargs);
    Matrix_init((PyObject*)m, args, kwargs);
    m->matrix->setZero();
    return (PyObject*)m;
}

static PyObject* Matrix_random(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // PyTypeObject* MatrixType = (PyTypeObject*)PyType_FromSpec(&matrix_spec);
    ModuleState* state = get_module_state(self);
    Matrix* m          = (Matrix*)Matrix_new(state->matrix_type, args, kwargs);
    Matrix_init((PyObject*)m, args, kwargs);
    m->matrix->setRandom();
    return (PyObject*)m;
}

static PyObject* Matrix_from(PyObject* self, PyObject* args)
{
    PyObject* data = nullptr;
    if (!PyArg_ParseTuple(args, "O", &data))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 1");
        return nullptr;
    }
    if (!PyList_Check(data))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 2");
        return nullptr;
    }
    int cols = PyList_GET_SIZE(data);
    if (cols <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 2");
        return nullptr;
    }
    PyObject* list = PyList_GET_ITEM(data, 0);
    if (!PyList_Check(list))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 3");
        return nullptr;
    }
    int rows               = PyList_GET_SIZE(list);
    Eigen::MatrixXd* p_mat = new Eigen::MatrixXd(rows, cols);
    for (int i = 0; i < cols; i++)
    {
        PyObject* list = PyList_GET_ITEM(data, i);
        if (!PyList_Check(list))
        {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 3");
            return nullptr;
        }
        int tmp = PyList_GET_SIZE(list);
        if (rows != tmp)
        {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. Each elements of it must be the same length.");
            return nullptr;
        }
        rows = tmp;

        for (int j = 0; j < rows; j++)
        {
            PyObject* num = PyList_GET_ITEM(list, j);
            if (!PyFloat_Check(num))
            {
                PyErr_SetString(PyExc_ValueError, "Every elements of the matrix must float.");
                return nullptr;
            }
            (*p_mat)(i, j) = ((PyFloatObject*)num)->ob_fval;
        }
    }
    ModuleState* state = get_module_state(self);
    return ReturnMatrix(p_mat, state->matrix_type);
}

static PyMethodDef module_methods[] = {
    { "ones", (PyCFunction)Matrix_ones, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values one." },
    { "zeros", (PyCFunction)Matrix_zeros, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values zero." },
    { "random", (PyCFunction)Matrix_random, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with random values" },
    { "matrix_from", (PyCFunction)Matrix_from, METH_VARARGS, "Return a new matrix with given values" },
    { NULL, NULL, 0, NULL }
};

static int module_traverse(PyObject* module, visitproc visit, void* arg)
{
    ModuleState* state = get_module_state(module);
    Py_VISIT(state->matrix_type);
    return 0;
}
static int module_clear(PyObject* module)
{
    ModuleState* state = get_module_state(module);
    Py_CLEAR(state->matrix_type);
    return 0;
}

static void module_free(void* module)
{
    module_clear((PyObject*)module);
}
/* Initialize this module. */
static int module_exec(PyObject* module)
{
    ModuleState* state = get_module_state(module);
    state->matrix_type = (PyTypeObject*)PyType_FromModuleAndSpec(module, &matrix_spec, nullptr);
    if (PyModule_AddObjectRef(module, "matrix", (PyObject*)state->matrix_type) < 0)
    {
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
    //     Py_DECREF(matrix_type);  //Failed to register, release the class object
    //     return -1;
    // }
    // Py_DECREF(matrix_type);
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    return 0;
}

static PyModuleDef_Slot module_slots[] = { { Py_mod_exec, (void*)module_exec },
#if PY_VERSION_HEX >= 0x030D0000
                                           { Py_mod_gil, Py_MOD_GIL_NOT_USED },
#endif
                                           { 0, NULL } };
PyDoc_STRVAR(
    module_doc,
    "Implementation module for SSL socket operations.  See the socket module\n\
for documentation.");

// static struct PyModuleDef module = {
//     .m_base = PyModuleDef_HEAD_INIT, .m_name = "matrix", .m_doc = module_doc, .m_size = sizeof(ModuleState), .m_methods = module_methods, .m_slots = module_slots, .m_traverse = module_traverse, .m_clear = module_clear, .m_free = module_free
// };

PyMODINIT_FUNC PyInit_matrix(void)
{
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
