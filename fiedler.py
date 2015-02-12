#!/usr/bin/python3
import networkx as nx
import scipy.linalg as la
import random
'''Check if generalized fiedler's theorem is true'''


def countsign(z):
    counter = 0
    if z[0] >= 0:
        sign = 1
    else:
        sign = -1
    for i in range(len(z)-1):
        if z[i+1]*sign < 0:
            counter += 1
            if z[i+1] >= 0:
                sign = 1
            else:
                sign = -1
    return counter


def general_fiedler_path(n, trials, k):
    print('k=' + str(k))
    G = nx.path_graph(n)
    L = nx.laplacian_matrix(G).toarray();
    (w, v) = la.eigh(L)
    flag = 0
    for x in range(trials):
        z = random.uniform(-5, 5)*v[:, 0]
        for i in range(0, k-1):
            z += random.uniform(-5, 5)*v[:, i+1]
        counter = countsign(z)
        if counter > k-1:
            print(z)
            print(counter)
            flag = 1
    if flag==0:
        print('None found')


def general_fiedler(G, trials, k):
    print('k='+str(k))
    L = nx.laplacian_matrix(G).toarray()
    (w, v) = la.eigh(L)
    flag = 0
    for i  in range(trials):
        z = random.uniform(-5, 5)*v[:, 0]
        for i in range(0, k-1):
            z += random.uniform(-5, 5)*v[:, i+1]
        y = threshold_cut(z)
        H = G.subgraph(y)
        n = nx.number_connected_components(H)
        if n > k-1:
            flag = 1
            print('number of components:' + str(n))
    if flag == 0:
        print('not found')


def threshold_cut(z):
    '''Output set W = {i : z_i \ge 0}'''
    y = []
    for i in range(len(z)):
        if z[i]>=0:
            y.append(i)
    return y


if __name__ == '__main__':
    n = 10
    k = 5
    trials = 10
    G = nx.path_graph(3000)
    general_fiedler(G, trials, k)
    # general_fiedler_path(n, trials, k)
