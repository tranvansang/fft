import numpy as np
import cmath
import sys


def calc_term(n, i, j):
    return cmath.exp(-1j * 2 * cmath.pi / n * ((i * j) % n))


def dft(a):
    n = len(a)
    mat = np.matrix([[1j for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            mat[i, j] = calc_term(n, i, j)
    return (mat * np.matrix(a).transpose()).transpose().tolist()[0]


def is_power2(n):
    return n and not n & (n - 1)


def fft_loop(a, depth, ind):
    n = len(a)
    if 1 << depth >= n:
        return [a[ind]]
    e = fft_loop(a, depth + 1,  ind)
    o = fft_loop(a, depth + 1,  ind | (1 << depth))
    n1 = n >> depth
    n2 = n1 >> 1
    ret = [0 for _ in range(n1)]
    for i in range(n2):
        term = calc_term(n, i << depth, 1)
        ret[i] = e[i] + term * o[i]
        ret[i + n2] = e[i] - term * o[i]
    return ret


def fft(a):
    if len(a) == 0:
        return []
    if not is_power2(len(a)):
        print("input len must be power of 2", file=sys.stderr)
        return []
    return fft_loop(a, 0,  0)


def main():
    a = [0.1, 0.5, 0.2, -.4, .3, .4, -.1, .5]
    print(np.fft.fft(a))
    print(dft(a))
    print(fft(a))


main()
