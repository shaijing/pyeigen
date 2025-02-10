# matrix
cpp_flag = []
mkl = dependency('mkl-dynamic-ilp64-iomp', required: false, method: 'pkg-config')
if mkl.found()
    cpp_flag += '-DUSE_MKL'
endif
eigen3 = dependency('eigen3')
numpy = dependency('numpy')
py.extension_module(
    'matrix',
    sources: ['matrix.h', 'matrix.cpp'],
    cpp_args: cpp_flag,
    dependencies: [py_dep, mkl, eigen3, numpy],
    install: true,
    subdir: 'pyeigen',
)

py.install_sources(
    [
        '__init__.py',
        'matrix.pyi',
    ],
    subdir: 'pyeigen',
)