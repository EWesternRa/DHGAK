from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


class ClusteringMethod:
    def __init__(self, method_name, n_clusters, kmeans_init='random', random_state=0, max_iter=100):
        self.method_name = method_name
        self.n_clusters = n_clusters
        self.kmeans_init = kmeans_init
        self.random_state = random_state
        self.max_iter = max_iter
        
    
    def fit(self, X, y=None):
        """
        fit clustering model
        X: array-like, shape (n_samples, n_features)
        """
        if self.method_name == 'k-means':
            self.model = KMeans(n_clusters=self.n_clusters, init=self.kmeans_init,
                                max_iter=self.max_iter, random_state=self.random_state)
        elif self.method_name == 'GMM':
            self.model = GaussianMixture(n_components=self.n_clusters, init_params=self.kmeans_init,
                                            max_iter=self.max_iter, random_state=self.random_state)
        elif self.method_name == 'DBSCAN':
            self.model = DBSCAN(eps=0.3, min_samples=5)
        else:
            raise NotImplementedError(f'Clustering method {self.method_name} not implemented.')
        self.model.fit(X)
        if self.method_name == 'DBSCAN':
            if -1 in self.model.labels_:
                # labels +1
                self.model.labels_ += 1
            self.n_clusters = len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0) # number of clusters, ignoring noise if present
        return self.model
        
    def predict(self, X):
        """
        predict cluster labels
        X: array-like, shape (n_samples, n_features)
        """
        if not hasattr(self, 'model'):
            raise Exception('model not fitted yet.')
        
        return self.model.predict(X)
    
    def fit_predict(self, X, y=None):
        """
        fit clustering model and predict cluster labels
        X: array-like, shape (n_samples, n_features)
        """
        self.fit(X, y)
        if self.method_name == 'DBSCAN':
            return self.model.labels_
        else:
            return self.model.predict(X)
    
    