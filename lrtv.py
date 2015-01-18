#!/usr/bin/env python3
import scipy as sp
import scipy.linalg as la


def laplacian(A):
    """Compute the normalized laplacian of an adjacency matrix"""
    d = sp.sum(A, axis=1)
    D = sp.diag(d)
    sqrt_d = [x**-0.5 for x in d]
    sqrt_D = sp.diag(sqrt_d)
    return sp.dot(sp.dot(sqrt_D, D-A), sqrt_D)


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


def cheeger_sweep(A, h, threshold):
    sets = []
    for i in range(len(h)):
        S = set()
        indices = [j[0] for j in sorted(enumerate(h[i]), key=lambda x:x[1])]
        nz_indices = [j for j in indices if h[i][j] != 0]
        min_sparsity = 2
        sparsest_set = {}
        for i in range(len(nz_indices)):
            S.add(nz_indices[i])
            if(sparsity(A, S) < min_sparsity):
                min_sparsity = sparsity(A, S)
                sparsest_set = S
        if(min_sparsity <= threshold):
            sets.append((sparsest_set, sparsity(A, sparsest_set)))
    return sets


if __name__ == '__main__':
    k = int(input('k = '))
    f = input('input file:')
    input_file = open(f, 'r')
    A = [y.strip('\n').rstrip(' ') for y in list(input_file)]
    A = [[float(z) for z in y.split(' ')] for y in A]
    c = float(input('constant:'))
    print('Graph:')
    print(A)
    L = laplacian(A)
    print('Laplacian:')
    print(L)
    (w, v) = spectral_projection(L, k)
    print('Eigenvalues: ')
    print(w)
    threshold = c*(w[0]*sp.log(k))
    print('threshold:')
    print(threshold)
    print('Eigenvectors: ')
    print(v.T)
    h = randomized_rounding(v, k)
    print('h:')
    print(h)
    print('Cheeger sweep sets:')
    print(cheeger_sweep(A, h, threshold))
