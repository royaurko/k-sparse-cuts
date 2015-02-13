import scipy as sp
import matplotlib.pyplot as plt
import math
import random
import fiedler


def fourier(n, k):
    p = math.pi
    a = []
    A = [[0 for x in range(k)] for x in range(n)]
    for i in range(0, k):
        a.append(random.uniform(-5, 5))
    for i in range(n):
        for j in range(k):
            A[i][j] = math.cos(j*p*(2*i+1)/(2*n))
    A = sp.matrix(A)
    v = sp.dot(A, a)
    y = sp.ravel(v)
    sign = fiedler.countsign(y)
    flag = 0
    if sign > k-1:
        print(y)
        flag = 1
    # plt.plot(range(n), y, linewidth=2)
    # plt.savefig('fourier-sign-changes')
    return flag


def experiment(n, trials):
    flag = 0
    for k in range(1, n):
        for i in range(trials):
            if fourier(n, k) > 0:
                flag = 1
    if flag == 0:
        print('not found')


if __name__ == '__main__':
    n = 100
    trials = 100
    experiment(n, trials)
