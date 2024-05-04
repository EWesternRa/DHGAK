import os
import random

import pandas as pd
import numpy as np
import networkx as nx
from sklearn import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection._validation import _fit_and_score
from sklearn.svm import SVC
from tqdm import tqdm

from .slices import get_graphs_slices

def load_data(dataset, data_path='datasets'):
    print('Loading {} dataset...'.format(dataset))
    adj_file = os.path.join(data_path, dataset, dataset + '_A.txt')
    graph_id_file = os.path.join(data_path, dataset, dataset + '_graph_indicator.txt')
    graph_label_file = os.path.join(data_path, dataset, dataset + '_graph_labels.txt')
    node_label_file = os.path.join(data_path, dataset, dataset + '_node_labels.txt')
    node_attr_file = os.path.join(data_path, dataset, dataset + '_node_attributes.txt')
    edge_label_file = os.path.join(data_path, dataset, dataset + '_edge_labels.txt')
    edge_attr_file = os.path.join(data_path, dataset, dataset + '_edge_attributes.txt')

    if not os.path.exists(adj_file):
        raise FileNotFoundError(f'Dataset {dataset} not found.')
        return None
    
    has_label = False
    has_attr = False
    has_edge_attr = False
    has_edge_label = False
    if os.path.exists(adj_file):
        adj = pd.read_csv(adj_file, header=None, index_col=None).values
    if os.path.exists(graph_id_file):
        graph_indicator = pd.read_csv(graph_id_file, header=None, index_col=None).values
    if os.path.exists(graph_label_file):
        graph_label = pd.read_csv(graph_label_file, header=None, index_col=None).values
    if os.path.exists(node_label_file):
        node_label = pd.read_csv(node_label_file, header=None, index_col=None).values
        has_label = True
    if os.path.exists(node_attr_file):
        node_attr = pd.read_csv(node_attr_file, header=None, index_col=None).values
        has_attr = True
    if os.path.exists(edge_label_file):
        edge_label = pd.read_csv(edge_label_file, header=None, index_col=None).values
        has_edge_label = True
    if os.path.exists(edge_attr_file):
        edge_attr = pd.read_csv(edge_attr_file, header=None, index_col=None).values
        has_edge_attr = True
    graphs = []
    num_graph = np.max(graph_indicator)
    edge_ind = 0
    for i in range(num_graph):
        g = nx.Graph()
        g.graph['label'] = graph_label[i][0]
        g.graph['id'] = i
        
        # add nodes
        node_indicator = np.where(graph_indicator == i+1)[0]
        node_indicator = node_indicator + 1    # 节点id从1开始
        for node in node_indicator:
            g.add_node(node)
            if has_label:
                g.nodes[node]['label'] = node_label[node-1][0]
            if has_attr:
                g.nodes[node]['attr'] = node_attr[node-1]
        
        # add edges
        while edge_ind < len(adj):
            edge = adj[edge_ind]
            if edge[0] in node_indicator and edge[1] in node_indicator:
                g.add_edge(edge[0], edge[1])
                # if has_edge_label:
                #     g.edges[edge[0], edge[1]]['label'] = edge_label[edge_ind][0]
                if has_edge_attr:
                    g.edges[edge[0], edge[1]]['attr'] = edge_attr[edge_ind][0]
                edge_ind += 1
            else:
                break
        
        # relabel to 0
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        if not has_label:
            # if no node labels, use degree
            node_labels = [g.degree[node] for node in g.nodes()]
            nx.set_node_attributes(g, dict(zip(g.nodes(), node_labels)), 'label')
        
        graphs.append(g)
        
    return graphs


def custom_grid_search_cv_kernel(model, param_grid, precomputed_kernels, y, cv=5, random_state=None):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    if random_state is not None:
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = []  # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score),
                                    train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc.get('test_scores'))
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    # print(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]

def crossvalidation_kernel(kernel, y, n_splits=10, random_state=0, gridsearch=True, crossvalidation=True):
    param_grid = [{
        'C': np.logspace(-2, 4, 7)
    }]
    
    kernel_matrices = [kernel]
        
    acc = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Hyperparam logging
    best_C = []
    print('training and gridsearch, fold:')
    for train_index, test_index in tqdm(cv.split(kernel_matrices[0], y)):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        # Gridsearch
        if crossvalidation:
            if gridsearch:
                max_iter = 1000000
                # max_iter = -1
                gs, best_params = custom_grid_search_cv_kernel(SVC(kernel='precomputed', cache_size=50, max_iter=max_iter,),
                                                        param_grid, K_train, y_train, cv=5, random_state=random_state)
                # Store best params
                C_ = best_params['params']['C']
                y_pred = gs.predict(K_test[best_params['K_idx']])
            else:
                gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
                y_pred = gs.predict(K_test[0])
                C_ = 100
            best_C.append(C_)
        else:
            gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            C_ = 100
            best_C.append(C_)
            acc.append(accuracy_score(y_test, y_pred))
            break

        acc.append(accuracy_score(y_test, y_pred))

    # ---------------------------------
    # Printing and logging
    # ---------------------------------
    if crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
            np.mean(acc) * 100,
            np.std(acc) * 100))
    else:
        print('Test accuracy: {:2.2f} %'.format(np.mean(acc) * 100))
        
    return acc, best_C


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def load_data_save_slices(dataset, b, H, data_path, load_slices_encoding, save_slices_encoding, node_centrality_path):
    load_file_flag = False

    if load_slices_encoding:
        file_path = f'./slices/{dataset}/slices-H{str(H)}b{str(b)}.npy'
        if not os.path.exists(file_path):
            load_file_flag = False
            print(f'file {file_path} not exists, generate slices from dataset.')
        else:
            load_file_flag = True
            nodes_slices = np.load(file_path, allow_pickle=True)
            y = np.load(f'./slices/{dataset}/y.npy', allow_pickle=True)
            num_nodes = np.load(f'./slices/{dataset}/num_nodes.npy', allow_pickle=True)
            print(f'load slice encoding done!')
    
    if not load_file_flag:
    
        graphs = load_data(dataset, data_path)
        
        y = np.array([graph.graph['label'] for graph in graphs])
    
        # get the slice encodings
        if not os.path.exists(node_centrality_path):
            os.makedirs(node_centrality_path)
        node_centrality_path = node_centrality_path + f'/{dataset}-node_centrality.npy'
        graphs_slices, num_nodes = get_graphs_slices(graphs, H, b, node_centrality_path,)
        
        nodes_slices = []
        for kneigs_labels in graphs_slices:
            nodes_slices.extend(kneigs_labels)
        
        if save_slices_encoding:
            file_path = f'./slices/{dataset}/slices-H{str(H)}b{str(b)}.npy'
            if not os.path.exists('./slices'):
                os.mkdir('./slices')
            if not os.path.exists(f'./slices/{dataset}'):
                os.mkdir(f'./slices/{dataset}')
            if not os.path.exists(file_path):   
                np_matrix = np.array(nodes_slices, dtype=object)
                np.save(file_path, np_matrix)   # save slices encoding
            num_nodes_path = f'./slices/{dataset}/num_nodes.npy'
            if not os.path.exists(num_nodes_path):
                np.save(num_nodes_path, np.array(num_nodes))    # save each graph's node number
            y_path = f'./slices/{dataset}/y.npy'
            if not os.path.exists(y_path):
                np.save(y_path, np.array(y))    # save each graph's label
            print(f'save slice encoding, num_nodes, y done!')
            
    return nodes_slices, y, num_nodes

def get_label2id(sentences):
    """
    Get label to id dictionary from sentences
    :param sentences: list of list of labels
    :return: label2id: dict
    """
    all_node_labels = set()
    for node_labels in sentences:
        for label in node_labels:
            all_node_labels = all_node_labels.union(set(label))
    # label to id
    label2id = dict(zip(all_node_labels, range(len(all_node_labels))))
    return label2id
