import numpy as np
from tqdm import tqdm
from .clustering_methods import ClusteringMethod
from joblib import Parallel, delayed

def dgak(slice_embs, num_nodes, cluster_methods, T, cluster_factor, cluster_jobs, random_state, max_iter=100, kmeans_init='random'):
    """
    obtain the feature maps of DGAK
    """
    def perform_iteration(method_name, n_clusters, kmeans_init, random_state, max_iter, slice_embs, num_nodes):
        method = ClusteringMethod(method_name, n_clusters, kmeans_init, random_state, max_iter)
        labels = method.fit_predict(slice_embs)
        cluster_indicators = np.eye(method.n_clusters)[labels]  # cluster indicators
        cluster_indicators = np.split(cluster_indicators, np.cumsum(num_nodes)[:-1])    # split into graphs
        feature_maps = [np.mean(ci, axis=0) for ci in cluster_indicators]   # feature maps
        return feature_maps
        
    iterations = []
    n_clusters = int(cluster_factor * len(num_nodes))
    cluster_seed = random_state
    for cm in cluster_methods:
        for _ in range(T):
            iterations.append((cm, n_clusters, kmeans_init, cluster_seed, max_iter, slice_embs, num_nodes))
            cluster_seed += 1

    feature_maps = Parallel(n_jobs=cluster_jobs)(delayed(perform_iteration)(*iteration) for iteration in iterations)
    # concat feature maps of different T and different clustering methods
    feature_maps = np.concatenate(feature_maps, axis=1)
        
    # divide by sqrt(T*len(cm))
    graph_featuremaps = feature_maps / np.sqrt(T * len(cluster_methods))
    return graph_featuremaps

def dhgak(slices_embs, num_nodes, cluster_methods, T, cluster_factor, cluster_jobs, random_state, max_iter=100, kmeans_init='random'):
    """
    obtain the kernel matrix of DHGAK
    """
    feature_maps = []
    dgak_seed = random_state
    for slice_embs in tqdm(slices_embs, desc='DHGAK'):
        feature_maps_dgak = dgak(slice_embs, num_nodes, cluster_methods, T, cluster_factor,
                                 cluster_jobs, dgak_seed, kmeans_init=kmeans_init, max_iter=max_iter)
        # concatenate the feature maps of different t and different clustering methods
        feature_maps.append(feature_maps_dgak)
        dgak_seed += 1
        
    feature_maps = np.concatenate(feature_maps, axis=1)
    kernel = feature_maps @ feature_maps.T
    diagonal_elements = np.diag(kernel)
    kernel /= np.sqrt(np.outer(diagonal_elements, diagonal_elements))
    return kernel
