import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class PCAGuidedKMeans: 
    def __init__(self, num_clusters, num_components = None, random_state = 40):
        self.num_clusters = num_clusters
        self.num_components = num_components if num_components else num_clusters
        self.random_state = random_state 
        self.pca_model = None
        self.kmeans_model = None 

    def pca_dimension_reduction(self, X): 

        """ Performs PCA on the input data 
            and returns the reduced data. """
        
        pca = PCA(n_components=self.num_components)
        X_reduced = pca.fit_transform(X)
        self.pca_model = pca 

        return X_reduced

    def kmeans_clustering(self, X):

        """Performs K-means clustering on the reinput data 
            and constructs the H matrix."""
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Construct the H matrix
        H = np.zeros((X.shape[0], self.num_clusters))
        for i in range(X.shape[0]):
            H[i, labels[i]] = 1  # Assign 1 to the column corresponding to the cluster of sample i
        
        self.kmeans_model = kmeans
        return kmeans, H       

    def kmeans_with_r1_initialization(self, X):

        """Performs K-means clustering with R1 initialization."""

        n_samples = X.shape[0]
        labels_r1 = np.random.randint(0, self.num_clusters, size=n_samples)

        # Compute initial centroids based on R1 initialization
        initial_centroids = np.array([X[labels_r1 == k].mean(axis=0) for k in range(self.num_clusters)])
        initial_centroids = np.nan_to_num(initial_centroids)  # Ensure no NaN values

        kmeans = KMeans(n_clusters=self.num_clusters, init=initial_centroids, n_init=1, random_state=self.random_state)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Construct the H matrix
        H = np.zeros((n_samples, self.num_clusters))
        for i in range(n_samples):
            H[i, labels[i]] = 1  # Assign 1 to the column corresponding to the cluster of sample i

        self.kmeans_model = kmeans
        return kmeans, H
    
    def compute_centroids(self, X, H):

        """Calculates the centroids of the clusters based on the H matrix."""

        num_clusters = H.shape[1]
        centroids = np.zeros((num_clusters, X.shape[1]))

        for j in range(num_clusters):
            indices = H[:, j] == 1
            cluster_points = X[indices]

            if len(cluster_points) > 0:
                centroids[j] = cluster_points.mean(axis=0)

        return centroids
    
    def kmeans_with_initial_centroids(self, X, initial_centroids):

        """Performs K-means clustering using specified initial centroids."""

        kmeans = KMeans(n_clusters=len(initial_centroids), init=initial_centroids, n_init=1)
        kmeans.fit(X)
        self.kmeans_model = kmeans
        return kmeans
    