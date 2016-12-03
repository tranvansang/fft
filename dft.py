import numpy as np
import time
import random
import matplotlib.pyplot as plt
import cmath
import sys


def calc_term(n, k):
    return cmath.exp(-1j * 2 * cmath.pi / n * (k % n))


def dft(a):
    n = len(a)
    mat = np.matrix([[1j for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            mat[i, j] = calc_term(n, i * j)
    return (mat * np.matrix(a).transpose()).transpose().tolist()[0]


def is_power2(n):
    return n and not n & (n - 1)


def bit_rev(x, l):
    return sum(1 << (l - 1 - i) for i in range(l) if (x >> i) & 1)


def bit_reverse(a):
    l = (len(a) - 1).bit_length()
    pos = [bit_rev(n, l) for n in range(len(a))]
    return [a[i] for i in pos]


def fft_loop(a, l, pos):
    l1 = l >> 1
    if l > 1:
        fft_loop(a, l1,  pos)
        fft_loop(a, l1,  pos + l1)
    for i in range(l1):
        ide = i + pos
        ido = i + pos + l1
        term = calc_term(l, i) * a[ido]
        a[ido] = a[ide] - term
        a[ide] += term


def fft(a):
    if len(a) == 0:
        return []
    if not is_power2(len(a)):
        print("input len must be power of 2", file=sys.stderr)
        return []
    a_copy = bit_reverse(a[:])
    fft_loop(a_copy, len(a_copy),  0)
    return a_copy


def easy_test():
    a = [0.1, 0.5, 0.2, -.4, .3, .4, -.1, .5]
    a = [0.1, 0.5, 0.2, -.4]
    print(np.fft.fft(a))
    print(dft(a))
    print(fft(a))


def gen_data(n):
    ret = []
    for _ in range(n):
        ret.append(random.uniform(-10., 10.))
    return ret


def fft_time(data):
    t = time.time()
    fft(data)
    return time.time() - t


def dft_time(data):
    t = time.time()
    dft(data)
    return time.time() - t


def main():
    x = []
    y1 = []
    y2 = []
    for i in range(10):
        x.append(1 << i)
        data = gen_data(1 << i)
        y1.append(fft_time(data))
        y2.append(dft_time(data))
    plt.plot(x, y1, "r")
    plt.plot(x, y2, "g")
    plt.show()

main()
