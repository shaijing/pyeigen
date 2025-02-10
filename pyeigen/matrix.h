#ifndef MATRIX_H
#define MATRIX_H
#ifdef USE_MKL
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif
#include <Eigen/Dense>
#include <Python.h>
#include <numpy/arrayobject.h>
#define CHECK_ALLOC(ptr)                                                                           \
    do {                                                                                           \
        if ((ptr) == NULL) {                                                                       \
            Py_DECREF(self);                                                                       \
            return NULL;                                                                           \
        }                                                                                          \
    } while (0)

template <int NPY_TYPE> struct NumpyTypeMap;
template <> struct NumpyTypeMap<NPY_INT> {
    using type = int;
};
template <> struct NumpyTypeMap<NPY_LONGLONG> {
    using type = long long;
};

template <> struct NumpyTypeMap<NPY_FLOAT> {
    using type = float;
};
template <> struct NumpyTypeMap<NPY_DOUBLE> {
    using type = double;
};

using MATRIX_DEFAULT_TYPE = double;
template <typename T>
using MatrixX   = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix    = MatrixX<MATRIX_DEFAULT_TYPE>;
using MatrixPtr = Matrix*;
struct MatrixObject {
    PyObject_HEAD

        int ndim;
    int size;
    PyObject* shape      = nullptr;
    PyObject* device     = nullptr;
    PyArray_Descr* dtype = nullptr;

    // Eigen::MatrixXD* matrix = nullptr;
    Matrix* matrix = nullptr;
};

/// Module state
struct ModuleState {
    PyTypeObject* matrix_type;
};

#endif  // MATRIX_H