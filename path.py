import networkx as nx
import scipy as sp
import scipy.linalg as la
import random


if __name__ == '__main__':
    n = 2000
    k = 500
    G = nx.path_graph(n)
    L = nx.laplacian_matrix(G).toarray()
    (w, v) = la.eigh(L)
    trials = 100
    print('k='+str(k))
    for x in range(trials):
        z = random.uniform(-5, 5)*v[:,0]
        for i in range(0,k-1):
            z += random.uniform(-5,5)*v[:,i+1]
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
        if counter > k-1:
            print(z)
            print(counter)
