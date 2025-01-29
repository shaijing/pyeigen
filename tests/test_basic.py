import pytest


def test_matrix():
    from pyeigen import ones, zeros, random, matrix_from

    m1 = ones(3, 3)
    m2 = zeros(3, 3)
    m3 = random(3, 3)
    m4 = matrix_from([[1.0, 2.0], [2.0, 2.0]])


def test_matrix_op():
    from pyeigen import ones, zeros, random, matrix_from

    m1 = ones(3, 3)
    m2 = zeros(3, 3)
    print((m1 + m2).data)
    print((m1 - m2).data)
    print((m1 * m2).data)
