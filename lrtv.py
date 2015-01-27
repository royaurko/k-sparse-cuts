#!/usr/bin/env python3
import os
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
    """Generate k iid spherical k-dim Gaussians g_1, ..., g_k"""
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
                tmp.append(sp.dot(v[j], v[j]))
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
        indices = [j[0] for j in sorted(enumerate(h[i]), key=lambda x: abs(x[1]))]
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
    k_cuts = sorted(sets, key=lambda x: x[2], reverse=False)
    return k_cuts


def lrtv(A, v, k, lambda_k, trials, f):
    """Call cheeger_sweep for a few number of trials"""
    k_cuts_list = []
    for i in range(0, trials):
        k_cuts = cheeger_sweep(A, v, k, lambda_k, f)
        k_cuts_list.append(k_cuts)
    return k_cuts_list


def hypercube():
    '''Generates a hypercube'''
    n = int(input('Number of dimensions: '))
    G = nx.hypercube_graph(n)
    return G


def random_graph():
    '''Erdos-Renyi G(n,p) graph'''
    n = int(input('Number of vertices'))
    p = float(input('probability: '))
    G = nx.fast_gnp_random_graph(n, p)
    return G


def random_regular():
    '''Generates a random regular graph'''
    n = int(input('Number of nodes: '))
    d = int(input('degree: '))
    G = nx.random_regular_graph(d, n)
    return G


def complete_graph():
    '''Generates a complete graph'''
    n = int(input('Number of nodes: '))
    G = nx.complete_graph(n)
    return G


def complete_bipartite():
    '''Generates a complete bipartite graph'''
    n = int(input('Number of nodes in partition 1: '))
    m = int(input('Number of nodes in partition 2: '))
    G = nx.complete_bipartite_graph(n, m)
    return G


def draw(G):
    '''Draws the graph and saves it as a png'''
    fname = input('save picture as: ')
    fname += '.png'
    nx.draw(G)
    plt.savefig(fname)


def plot(k_cuts_list, plotname):
    scale = 100
    for i in range(len(k_cuts_list)):
        y_data = [y for (x, y, z) in k_cuts_list[i]]
        x_data = [x*scale for x in range(len(y_data))]
        plt.plot(x_data, y_data, linewidth=2.0)


def generate_grid_graph():
    '''Generates k cuts for grid graphs'''
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
    plotname = gridfname + 'plot'
    plot(k_cuts_list, plotname)
    plt.savefig(plotname)
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
    '''Generates k cuts for cartesian product of a path and a binary tree'''
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
    plotname = prodfname + 'plot'
    plot(k_cuts_list, plotname)
    for i in range(len(k_cuts_list)):
        k_cuts = k_cuts_list[i]
        tmp_str = list(map(str, k_cuts))
        tmp_str = ' '.join(tmp_str)
        tmp_str += '\n\n'
        tmp_str = tmp_str.encode('utf-8')
        prodfile.write(tmp_str)


def generate_noisy_hypercube():
    '''Generates n dimensional noisy hypercube graph, with epsilon noise'''
    k = int(input('k for noisy hypercube: '))
    trials = int(input('number of trials: '))
    cubefname = input('output file:')
    cubefname = 'hard_instances/' + cubefname
    cubefile = open(cubefname, 'wb', 0)
    n = int(input('dimension of hypercube: '))
    nodes = 2**n
    epsilon = float(input('noise:'))
    G = nx.empty_graph(nodes)
    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue
            else:
                d = hamming_dist(u, v, n)
                w = epsilon**d
                G.add_edge(u, v, weight=w)
    A = nx.adjacency_matrix(G).toarray()
    L = nx.normalized_laplacian_matrix(G).toarray()
    (w, v) = spectral_projection(L, k)
    lambda_k = w[0]
    tmp_str = 'Noisy hypercube of dimension ' + str(n)
    tmp_str += ' with noise parameter ' + str(epsilon) + '\n'
    tmp_str += 'k = ' + str(k) + ', '
    tmp_str += 'trials = ' + str(trials) + '\n\n\n'
    tmp_str = tmp_str.encode('utf-8')
    cubefile.write(tmp_str)
    k_cuts_list = lrtv(A, v, k, lambda_k, trials, cubefile)
    plotname = cubefname + 'plot'
    plot(k_cuts_list, plotname)
    for i in range(len(k_cuts_list)):
        k_cuts = k_cuts_list[i]
        tmp_str = list(map(str, k_cuts))
        tmp_str = ' '.join(tmp_str)
        tmp_str += '\n\n'
        tmp_str = tmp_str.encode('utf-8')
        cubefile.write(tmp_str)


def generate_hard_instances():
    '''Generates cuts for known hard instances for k-sparse-cuts'''
    flag = int(input('1 for grid graph, 2 for product, 3 for noisy hypercube:'))
    if flag == 1:
        generate_grid_graph()
    elif flag == 2:
        generate_product_graph()
    elif flag == 3:
        generate_noisy_hypercube()
    else:
        # Else generate all
        generate_noisy_hypercube()
        generate_grid_graph()
        generate_product_graph()


def hamming_dist(x, y, n):
    '''Return hamming distance of two n bit integers'''
    tmp_str = '{0:0'
    tmp_str += str(n)
    tmp_str += 'b}'
    x = tmp_str.format(x)
    y = tmp_str.format(y)
    return sum(ch1 != ch2 for ch1, ch2 in zip(x, y))


if __name__ == '__main__':
    foldername = 'hard_instances'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    generate_hard_instances()
