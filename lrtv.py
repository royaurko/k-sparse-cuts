#!/usr/bin/env python3
import scipy as sp
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt


def spectral_projection(A, k):
    """Generate top k eigenvectors of A"""
    n = len(A)
    w, v = la.eigh(A, eigvals=(n-k, n-1))
    return (w, v)


def generate_gaussians(k):
    """Generate k spherical k-dimensional Gaussians g_1, ..., g_k"""
    mean = [0 for x in range(0, k)]
    covariance = sp.matrix(sp.identity(k), copy=False)
    g = []
    for i in range(0, k):
        tmp = sp.random.multivariate_normal(mean, covariance)
        g.append(tmp)
    return g


def randomized_rounding(v, k):
    """Compute the vectors h_1, ..., h_k as in LRTV"""
    g = generate_gaussians(k)
    h = []
    for i in range(0, k):
        tmp = []
        for j in range(len(v)):
            # compute the max
            inner_prd = [sp.dot(v[j], x) for x in g]
            i_max = inner_prd.index(max(inner_prd))
            if(i == i_max):
                tmp.append(inner_prd[i])
            else:
                tmp.append(0)
        h.append(tmp)
    return h


def sparsity(A, S):
    """Computes the sparsity of a cut S, where S is a set of vertex indices"""
    total = sp.sum(A)/2
    crossing = 0
    total_S = 0
    total_inside = 0
    for i in range(len(A)):
        if(i not in S):
            continue
        for j in range(len(A)):
            if(j not in S):
                crossing += A[i][j]
                total_S += A[i][j]
            else:
                total_inside += A[i][j]/2
                total_S += A[i][j]/2
    if(total_S > total - total_inside):
        # Take the minimum of w(S) and w(S_c), where S_c is the complement of S
        total_S = total - total_inside
    sparsity = crossing/total_S
    return sparsity


def cheeger_sweep(A, v, k, c, lambda_k):
    """Generate h and take best cut in some ck trials"""
    # repeat the algorithm for
    h = randomized_rounding(v, k)
    sets = []
    threshold = lambda_k * sp.log(k)
    threshold = threshold ** 0.5
    for i in range(0, k):
        indices = [j[0] for j in sorted(enumerate(h[i]), key=lambda x: x[1])]
        # Discard vertices with zero entries
        nz_indices = [j for j in indices if h[i][j] != 0]
        if not nz_indices:
            continue
        S = set()
        S.add(nz_indices[0])
        min_sparsity = sparsity(A, S)
        sparsest_set = S
        for i in range(2, len(nz_indices)):
            S.add(nz_indices[i])
            tmp_sparsity = sparsity(A, S)
            if(tmp_sparsity < min_sparsity):
                min_sparsity = tmp_sparsity
                sparsest_set = S
        # At this point we have the sparsest cheeger cut from h_i
        sets.append((sparsest_set, min_sparsity, min_sparsity/threshold))
    # Sort the k-cuts according to their min_sparsity/threshold ratio
    k_cuts = sorted(sets, key=lambda x: x[2], reverse=True)
    return k_cuts


def lrtv(A, v, k, c, lambda_k, trials):
    """Call cheeger_sweep for a few number of trials"""
    for i in range(0, trials):
        print('k-cuts:')
        print(cheeger_sweep(A, v, k, c, lambda_k))


if __name__ == '__main__':
    k = int(input('k = '))
    n = int(input('n = '))
    # fname = input('file:')
    # f = open(fname, 'r')
    # A = [y.strip('\n').rstrip(' ') for y in list(f)]
    # A = [[float(z) for z in y.split(' ')] for y in A]
    G = nx.fast_gnp_random_graph(n, 0.5)
    nx.draw(G)
    plt.savefig('fig.png')
    A = nx.adjacency_matrix(G).toarray()
    c = float(input('constant:'))
    trials = int(input('number of runs: '))
    L = nx.normalized_laplacian_matrix(G).toarray()
    (w, v) = spectral_projection(L, k)
    lambda_k = w[0]
    lrtv(A, v, k, c, lambda_k, trials)
