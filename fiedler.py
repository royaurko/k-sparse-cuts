import networkx as nx
import scipy as sp
import scipy.linalg as la
import random
import matplotlib.pyplot as plt


def keigenvectors(G, k):
    '''Return the bottom k eigenvectors of the Laplacian'''
    L = nx.laplacian_matrix(G).toarray()
    (w, v) = la.eigh(L, eigvals=(0, k-1))
    return v


def thresholdcut(z, t):
    '''Return the set W = { i : z_i \ge t}'''
    cut1 = [i for i in range(len(z)) if z[i] >= t]
    cut2 = [i for i in range(len(z)) if z[i] < t]
    return (cut1, cut2)


def edgesignchange(G, z, t):
    '''Return the number of edges (u, v) such that z_u*z_v < t'''
    counter = 0
    for edge in G.edges():
        if z[edge[0]]*z[edge[1]] < t:
            counter += 1
    return counter


def randomvector(v):
    '''Generate a random vector in the span of the columns of v'''
    multiplier = []
    k = len(v[0])
    for j in range(k):
        multiplier.append(random.uniform(0, 5))
    return sp.dot(v, multiplier)


def nonnegative(z):
    '''For a star graph if the central vertex is negative, counts the number of nonnegative entries'''
    if z[0] >= 0:
        return
    counter = 0
    for i in range(1, len(z)):
        if z[i] >= 0:
            counter += 1
    return counter


def general_fiedler(G, k, trials, plotname):
    '''Number of components when you apply the threshold cut on a random vector in the span of 1st k'''
    v = keigenvectors(G, k)
    print v
    flag = 1
    x_data = []
    y_data = []
    for i in range(trials):
        z = randomvector(v)
        (y1, y2) = thresholdcut(z, 0)
        H1 = G.subgraph(y1)
        n1 = nx.number_connected_components(H1)
        H2 = G.subgraph(y2)
        n2 = nx.number_connected_components(H2)
        if n1 < n2:
            n = n1
        else:
            n = n2
        x_data.append(i)
        y_data.append(n)
        if n > k-1:
            flag = 0
            print 'Number of components: ' + str(n)
            print 'z = ' + str(z)
    if flag:
        print 'Not found, number of components: ' + str(n)
    k_data = [k-1 for x in x_data]
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, k_data, linewidth=2)
    plt.axis([0, trials, 0, k+10])
    plt.savefig(plotname)


def search(G, trials, plotname):
    n = len(G.nodes())
    for k in range(2, n):
        general_fiedler(G, k, trials, plotname)


def doublethreshold(G, v):
    '''Given vector v check number of components in both sides of all possible thresholds'''
    n = len(v)
    u = [(v[i], i) for i in range(n)]
    u.sort()
    result = list()
    for i in range(1, n):
        tmp1 = u[:i]
        tmp2 = u[i:]
        tmp1 = [x[1] for x in tmp1]
        tmp2 = [x[1] for x in tmp2]
        H1 = G.subgraph(tmp1)
        H2 = G.subgraph(tmp2)
        n1 = nx.number_connected_components(H1)
        n2 = nx.number_connected_components(H2)
        result.append(max(n1, n2))
    return reduce(lambda x, y:max(x, y), result)
