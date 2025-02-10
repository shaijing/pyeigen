import time
import mymodule


def init_numpy_mkl():
    try:
        import os
        from pathlib import Path

        MKLROOT_ENV = os.environ.get("MKLROOT")
        if MKLROOT_ENV is None:
            raise EnvironmentError("MKLROOT environment variable not found!")
        MKLROOT = Path(MKLROOT_ENV)
        os.add_dll_directory((MKLROOT / "bin").as_posix())

    except Exception as e:
        raise RuntimeError(f"Failed to initialize NumPy with MKL: {e}")


init_numpy_mkl()
del init_numpy_mkl

print(mymodule.__doc__)
print(mymodule.__version__)
mymodule.hello()
print(mymodule.hello.__doc__)
print(mymodule.Complex)

a = mymodule.Complex(1, 2)
print(a.real)
print(a.imag)

import matrix

print(matrix.__version__)
print(matrix.matrix)

m1 = matrix.ones(3, 3)
m2 = matrix.ones(3, 3)
print((m1 + m2).data)
print((m1 - m2).data)
print((m1 * m2).data)
m3 = matrix.matrix_from([[1.0, 2.0], [2.0, 2.0]])

mat_a = matrix.random(1000, 1000)
mat_b = matrix.ones(1000, 1000)
mat_c = matrix.zeros(1000, 1000)
start_time = time.time()
_ = mat_a * mat_b
print("Time: ", time.time() - start_time)
