import os

import numpy as np
from scipy.sparse.linalg import eigsh
import networkx as nx

def slices_encoding(graph, H, b, node_centralities,):
    """
    Get the Slice encodings.
    H: maximum hops
    b: width
    node_centralities: node centralities
    """
    all_neigbors_labels = []
    all_bfs = []

    # BFS for each node with b-depth
    for node in graph.nodes():
        visited = [node]
        bfs_seq = [node]
        k_neis = [node]
        for _ in range(b):
            kth_neis = []
            while len(bfs_seq) != 0:
                v = bfs_seq.pop(0)
                v_neis = [n for n in graph.neighbors(v) if n not in visited]
                if len(v_neis) == 0:
                    continue
                visited.extend(v_neis)
                kth_neis.extend(v_neis)
            kth_neis = list(set(kth_neis))
            bfs_seq = kth_neis.copy()
            k_neis = k_neis + sorted(kth_neis, key=lambda x: node_centralities[x], reverse=True)
        all_bfs.append(k_neis)
            
    
    for node in graph.nodes():
        neigs = [node]
        neis_bfs_seq = [all_bfs[node]]
        old_neigs = neigs
        for _ in range(H):
            kth_neigbors = sorted(list(set([n for nei in old_neigs for n in graph.neighbors(nei) if n not in neigs])),
                                  key=lambda x: node_centralities[x], reverse=True)
            if len(kth_neigbors) == 0:
                break
            else:
                old_neigs = kth_neigbors.copy()
                neigs = neigs + kth_neigbors
                
                tmp = []
                for l in [all_bfs[n] for n in kth_neigbors]:
                    tmp = tmp + l
                neis_bfs_seq.append(tmp)
                
        
        # for less than H hops, repeat the last node
        if len(neis_bfs_seq) < H + 1:
            neis_bfs_seq = neis_bfs_seq + [neis_bfs_seq[-1] for _ in range(H + 1 - len(neis_bfs_seq))]
        
        # transform to labels
        k_neigbors_label = []
        for neigs in neis_bfs_seq:
            neigs_label = [graph.nodes[neig]['label'] for neig in neigs]
            k_neigbors_label.append(neigs_label)
        
        all_neigbors_labels.append(k_neigbors_label)

    return all_neigbors_labels

def get_graphs_slices(graphs, H, b, node_centrality_path,):
    """
    Obtain the Slice encodings for all graphs.
    node_centrality_path: path to save node centralities
    """
    graphs_slices = []

    # compute node centralities or load from file

        
    if os.path.exists(node_centrality_path):
        all_centralities = np.load(node_centrality_path, allow_pickle=True)
    else:
        all_centralities = []
        for graph in graphs:
            adj = nx.adjacency_matrix(graph).todense().astype(np.float32)
            _, v = eigsh(adj, k=1, which='LM')
            node_centralities = np.abs(v[:, 0])
            # normalize
            node_centralities = node_centralities / np.sum(node_centralities)
            all_centralities.append(node_centralities)

        all_centralities = np.array(all_centralities, dtype=object)
        np.save(node_centrality_path, all_centralities)
            
    for idx, graph in enumerate(graphs):
        graph_slice = slices_encoding(graph, H, b, all_centralities[idx])
        
        graphs_slices.append(graph_slice)

    return graphs_slices, [len(l) for l in graphs_slices]
