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


def cheeger_sweep(A, v, k, lambda_k, f):
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


def lrtv(A, v, k, lambda_k, trials, f):
    """Call cheeger_sweep for a few number of trials"""
    k_cuts_list = []
    for i in range(0, trials):
        k_cuts = cheeger_sweep(A, v, k, lambda_k, f)
        k_cuts_list.append(k_cuts)
    return k_cuts_list


def hypercube():
    n = int(input('Number of dimensions: '))
    G = nx.hypercube_graph(n)
    return G


def random_graph():
    n = int(input('Number of vertices'))
    p = float(input('probability: '))
    G = nx.fast_gnp_random_graph(n, p)
    return G


def random_regular():
    n = int(input('Number of nodes: '))
    d = int(input('degree: '))
    G = nx.random_regular_graph(d, n)
    return G


def complete_graph():
    n = int(input('Number of nodes: '))
    G = nx.complete_graph(n)
    return G


def complete_bipartite():
    n = int(input('Number of nodes in partition 1: '))
    m = int(input('Number of nodes in partition 2: '))
    G = nx.complete_bipartite_graph(n, m)
    return G


def draw(G):
    fname = input('save picture as: ')
    fname += '.png'
    nx.draw(G)
    plt.savefig(fname)


def generate_grid_graph():
    k = int(input('k for grid graph:'))
    trials = int(input('number of trials:'))
    gridfname = input('output file:')
    gridfname = 'hard_instances/' + gridfname
    gridfile = open(gridfname, 'wb', 0)
    n = int(input('Number of dimensions: '))
    d = []
    for i in range(0, n):
        tmp = int(input('Size of dimension ' + str(i+1) + ': '))
        d.append(tmp)
    G = nx.grid_graph(dim=d)
    A = nx.adjacency_matrix(G).toarray()
    L = nx.normalized_laplacian_matrix(G).toarray()
    (w, v) = spectral_projection(L, k)
    lambda_k = w[0]
    k_cuts_list = lrtv(A, v, k, lambda_k, trials, gridfile)
    tmp_str = 'Grid graph of dimension: ' + str(d) + '\n'
    tmp_str += 'k = ' + str(k) + ', '
    tmp_str += 'trials = ' + str(trials) + '\n\n\n'
    tmp_str = tmp_str.encode('utf-8')
    gridfile.write(tmp_str)
    for i in range(len(k_cuts_list)):
        k_cuts = k_cuts_list[i]
        tmp_str = list(map(str, k_cuts))
        tmp_str = ' '.join(tmp_str)
        tmp_str += '\n\n'
        tmp_str = tmp_str.encode('utf-8')
        gridfile.write(tmp_str)


def generate_product_graph():
    k = int(input('k for product of tree & path:'))
    trials = int(input('number of trials:'))
    prodfname = input('output file:')
    prodfname = 'hard_instances/' + prodfname
    prodfile = open(prodfname, 'wb', 0)
    h = int(input('height of the tree: '))
    H = nx.balanced_tree(2, h)
    n = int(input('number of nodes in path: '))
    G = nx.path_graph(n)
    T = nx.cartesian_product(G, H)
    A = nx.adjacency_matrix(T).toarray()
    L = nx.normalized_laplacian_matrix(T).toarray()
    (w, v) = spectral_projection(L, k)
    lambda_k = w[0]
    tmp_str = 'Cartesian product of balanced tree of height ' + str(h)
    tmp_str += ' and path of length ' + str(n-1) + '\n'
    tmp_str += 'k = ' + str(k) + ', '
    tmp_str += 'trials = ' + str(trials) + '\n\n\n'
    tmp_str = tmp_str.encode('utf-8')
    prodfile.write(tmp_str)
    k_cuts_list = lrtv(A, v, k, lambda_k, trials, prodfile)
    for i in range(len(k_cuts_list)):
        k_cuts = k_cuts_list[i]
        tmp_str = list(map(str, k_cuts))
        tmp_str = ' '.join(tmp_str)
        tmp_str += '\n\n'
        tmp_str = tmp_str.encode('utf-8')
        prodfile.write(tmp_str)


def generate_hard_instances():
    '''Generates cuts for known hard instances for k-sparse-cuts'''
    flag = int(input('1 for grid graph, 2 for product:'))
    if flag == 1:
        generate_grid_graph()
    elif flag == 2:
        generate_product_graph()
    else:
        generate_grid_graph()
        generate_product_graph()


if __name__ == '__main__':
    generate_hard_instances()
