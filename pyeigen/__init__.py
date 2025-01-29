def init_mkl():
    try:
        import os
        from pathlib import Path

        MKLROOT_ENV = os.environ.get("MKLROOT")
        if MKLROOT_ENV is None:
            raise EnvironmentError("MKLROOT environment variable not found!")
        MKLROOT = Path(MKLROOT_ENV)
        os.add_dll_directory(MKLROOT / "bin")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize MKL: {e}")


init_mkl()
del init_mkl

from .matrix import matrix
from .matrix import ones, zeros, random, matrix_from

__all__ = ["matrix", "ones", "zeros", "random", "matrix_from"]
