import networkx as nx
import numpy as np
import operator
import os
import dgl 
import torch
import pdb

def random_walk_sampler(G, n_sample=30, max_its=1000, seed=1, set_seed=False, Tm=None):

    if set_seed:
        np.random.seed(seed)

    if n_sample > len(G.nodes()):
      n_sample = len(G.nodes())

    if Tm is None:  
        A = nx.adj_matrix(G)
        A = A.todense()
        A = np.array(A, dtype = np.float64)
        # let's evaluate the degree matrix D
        Dinv = np.diag(1 / np.sum(A, axis=1))
        # ...and the transition matrix T
        Tm = np.dot(Dinv,A)

    nodes = list(G.nodes())
    # pick a random initial node
    start_idx = np.random.choice(np.arange(Tm.shape[0]), size=1)[0]
    start_node = nodes[start_idx]

    visited = set([start_node])
    visited_idx = [start_idx]

    its = 0

    while len(visited) < n_sample and its < max_its:
        next_idx = np.random.choice(np.arange(Tm.shape[0]), 
                                    size=1, p=Tm[visited_idx[-1], :])[0]
        next_node = nodes[next_idx]
        visited.add(next_node)
        visited_idx.append(next_idx)
        its += 1

    if its == max_its:
      print('Failed to sample enough nodes.')

    g_sub = G.subgraph(list(visited)).copy()
    g_sub = nx.convert_node_labels_to_integers(g_sub)
    clustering_coefs = nx.clustering(g_sub)
    clcfs = [clustering_coefs[n] for n in g_sub.nodes()]

    return g_sub, clcfs

def rw_from_edge_list_file(path, fname, n_sample):

    edges = np.loadtxt(
        os.path.join(path, fname), dtype=np.int32)
    G = nx.from_edgelist(edges)
    g_sub, clcfs = random_walk_sampler(G, n_sample=n_sample)

    return g_sub, clcfs

def make_net_with_max_degree_config_model(G, max_degree):

    degree_sequence = np.array(list(dict(G.degree()).values()))
    degree_sequence_trunc = np.minimum(degree_sequence, max_degree)
    if degree_sequence_trunc.sum() % 2 != 0:
        degree_sequence_trunc[np.argmin(degree_sequence_trunc)] += 1

    G_new = nx.configuration_model(degree_sequence_trunc)
    G_new = nx.Graph(G_new)

    return G_new

def make_barabasi_albert_network(n=10, m=2):
    G = nx.DiGraph(nx.barabasi_albert_graph(n, m))
    return dgl.DGLGraph(G)

def make_core_periphery(n=10, n_core=2, n_h=1):
    
    core = np.random.choice(np.arange(0, n), size=n_core, replace=False)
    labels = np.zeros(n, dtype=np.int64)
    labels[core] = 1
    
    h_type = np.random.choice(np.arange(0, n), size=n_h, replace=False)
    hl_type = np.zeros(n, dtype=np.int64)
    hl_type[h_type] = 1
    
    G = dgl.DGLGraph()
    G.add_nodes(n, data={'cp_label' : torch.tensor(labels), 'hl_label' : torch.tensor(hl_type)})

    for n1 in core:
        for n2 in core:
            if n1 != n2:
                G.add_edges(n1, n2)
                G.add_edges(n2, n1)
    
    for node in range(0, n):
        if node not in core:
            r = np.random.choice(core, size=1)
            G.add_edges(node, r)
            G.add_edges(r, node)
    
    return G, labels

def make_star(n=10):

    G = dgl.DGLGraph()

    G.add_nodes(n)
    src = torch.tensor(list(range(0, n)))
    center = np.random.randint(0, n)
    G.add_edges(src, center)
    G.add_edges(center, src)
    labels = np.zeros(n, dtype=np.int64)
    labels[center] = 1

    return G, labels

def make_degree_effect_graph(n=10):
    
    node_labels = np.random.permutation(np.arange(n))
    
    G = dgl.DGLGraph()
    G.add_nodes(n)
    
    G.add_edges(node_labels[0], node_labels[1])
    G.add_edges(node_labels[1], node_labels[0])
    
    G.add_edges(node_labels[1], node_labels[2])
    G.add_edges(node_labels[2], node_labels[1])
    
    for i in range(3, n):
        G.add_edges(node_labels[2], node_labels[i])
        G.add_edges(node_labels[i], node_labels[2])
        
    return G

def make_two_star(n=11):
    
    node_labels = np.random.permutation(np.arange(n))
    
    G = dgl.DGLGraph()
    G.add_nodes(n)
    
    cnt = 1
    
    while cnt < n:
        G.add_edges(node_labels[0], node_labels[cnt])
        G.add_edges(node_labels[cnt], node_labels[0])
        G.add_edges(node_labels[cnt], node_labels[cnt+1])
        G.add_edges(node_labels[cnt+1], node_labels[cnt])
        cnt += 2
        
    return G
    

def make_fb_nets(facebook_graph, facebook_TM, n=10, max_degree=None):

    G_nx, _ = random_walk_sampler(facebook_graph, n_sample=n, Tm=facebook_TM)

    if max_degree is not None:
        if max_degree < n:
            G_nx = restrict_max_degree(G_nx, max_degree)

    G = dgl.DGLGraph()
    G.from_networkx(G_nx)

    return G

def get_max_degree_node(G, nodes=None):
    if nodes is None:
        degree_list = [key for key in G.degree()]
    else:
        degree_list = [key for key in G.degree(nodes)]
    degree_list = sorted(degree_list, key=lambda x:x[1], reverse=True)
    max_degree = degree_list[0][1]
    max_node = degree_list[0][0]
    
    return max_degree, max_node

def restrict_max_degree(G, max_degree_limit):

    max_degree, max_node = get_max_degree_node(G)
    while max_degree > max_degree_limit:
        while max_degree > max_degree_limit:
            edges = G.edges(max_node)
            terminal_nodes = [e[1] for e in edges]
            _, edge_remove_node = get_max_degree_node(G, nodes=terminal_nodes)
            G.remove_edge(max_node, edge_remove_node)
            max_degree = max_degree - 1
        max_degree, max_node = get_max_degree_node(G)
    
    return G

    
