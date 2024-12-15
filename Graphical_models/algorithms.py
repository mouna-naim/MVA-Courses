"""
    This code implements reference clustering algorithms: KMeans++, PCA-guided KMeans,
    KMeans with R1 initialization, KMeans with R2 initialization, KMeans with HAC initialization,
    and KMeans on reduced data.
"""

# Necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from pca_kmeans import PCAGuidedKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


#KMeans++ algorithm
def kmeanspp_clustering(X, n_clusters, max_iter, run):
    """
    Applies KMeans++ and computes distortions and J_best(t) at each iteration.
    Displays the values of J_best(t) for each iteration.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1, n_init=1, random_state=run)
    distortions = []

    for iteration in range(1, max_iter + 1):
        kmeans.fit(X)  # Performs one iteration
        distortions.append(kmeans.inertia_)  # Records the distortion at this iteration
        
        # Calculate J_best(t) up to the current iteration
        j_best = calculate_j_best(distortions)
        
        # Prepare for the next iteration
        kmeans.max_iter += 1

    # Returns the final KMeans model and J_best(t) values
    return kmeans, calculate_j_best(distortions)

#PCA-guided KMeans algorithm
def pca_guided_clustering(X, n_clusters, max_iter, run):
    """
    Applies PCA-guided KMeans and computes distortions and J_best(t) at each iteration.
    Displays the values of J_best(t) for each iteration.
    """

    X = np.array(X)

    # Initial PCA-guided step
    pca_guided = PCAGuidedKMeans(num_clusters=n_clusters, random_state=run)
    X_reduced = pca_guided.pca_dimension_reduction(X)  # Dimensionality reduction
    kmeans, H = pca_guided.kmeans_clustering(X_reduced)  # Initial clustering on reduced data
    centroids = pca_guided.compute_centroids(X, H)  # Compute centroids
    kmeans = KMeans(n_clusters=len(centroids), init=centroids, n_init=1, max_iter=1, random_state=run)

    distortions = []

    # Loop through iterations
    for iteration in range(1, max_iter + 1):
        kmeans.fit(X)  # Performs one iteration
        distortions.append(kmeans.inertia_)  # Records the distortion at this iteration

        # Calculate J_best(t) up to the current iteration
        j_best = calculate_j_best(distortions)

        kmeans.max_iter += 1

    return kmeans, calculate_j_best(distortions)


#KMeans with R1 initialization
def kmeans_with_r1_initialization(X, n_clusters, max_iter, run):
    """
    Applies KMeans with R1 initialization and computes distortions and J_best(t) at each iteration.
    Displays the values of J_best(t) for each iteration.
    """
    X = np.array(X)

    # R1 initialization
    np.random.seed(run)
    random_labels = np.random.randint(0, n_clusters, size=X.shape[0])
    initial_centroids = np.array([X[random_labels == i].mean(axis=0) for i in range(n_clusters)])
    initial_centroids = np.nan_to_num(initial_centroids)

    # KMeans initialization
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, max_iter=1, n_init=1, random_state=run)

    distortions = []

    # Loop through iterations
    for iteration in range(1, max_iter + 1):
        kmeans.fit(X)  # Performs one iteration
        distortions.append(kmeans.inertia_)  # Records the distortion at this iteration

        # Calculate J_best(t) up to the current iteration
        j_best = calculate_j_best(distortions)

        kmeans.max_iter += 1

    return kmeans, calculate_j_best(distortions)

#KMeans with R2 initialization
def kmeans_with_r2_initialization(X, n_clusters, max_iter, run):
    """
    Applies KMeans with R2 initialization and computes distortions and J_best(t) at each iteration.
    Displays the values of J_best(t) for each iteration.
    """
    X = np.array(X)
    np.random.seed(run)
    initial_centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]  # R2 initialization
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, max_iter=1, n_init=1, random_state=run)

    distortions = []

    # Loop through iterations
    for iteration in range(1, max_iter + 1):
        kmeans.fit(X)  # Performs one iteration
        distortions.append(kmeans.inertia_)  # Records the distortion at this iteration

        # Calculate J_best(t) up to the current iteration
        j_best = calculate_j_best(distortions)

        kmeans.max_iter += 1

    return kmeans, calculate_j_best(distortions)

#KMeans with HAC initialization
def hac_initialization(X, n_clusters, max_iter, run):
    """
    Applies HAC initialization with calculation and display of J_best(t).
    """
    X = np.array(X)

    def hac_initialization(X, n_clusters):
        """
        HAC initialization.
        """
        X = np.array(X)
        hac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hac.fit_predict(X)
        centroids = []
        for cluster_label in range(n_clusters):
            cluster_points = X[labels == cluster_label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    # HAC initialization
    initial_centroids = hac_initialization(X, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, max_iter=1, n_init=1, random_state=run)

    distortions = []

    # Loop through iterations
    for iteration in range(1, max_iter + 1):
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

        # Calculate and display J_best(t)
        j_best = calculate_j_best(distortions)

        kmeans.max_iter += 1

    return kmeans, calculate_j_best(distortions)

# KMeans algorithm on reduced data
def pca_Kmeans(X, n_clusters, max_iter, run):
    """
    Applies PCA-guided KMeans and computes distortions and J_best(t) at each iteration.
    Displays the values of J_best(t) for each iteration.
    """

    X = np.array(X)

    # Initial PCA-guided step
    pca_guided = PCAGuidedKMeans(num_clusters=n_clusters, random_state=run)
    X_reduced = pca_guided.pca_dimension_reduction(X)  # PCA
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=1, n_init=1, random_state=run)
    distortions = []

    for iteration in range(1, max_iter + 1):
        kmeans.fit(X_reduced)  # Performs one iteration
        distortions.append(kmeans.inertia_)  # Records the distortion at this iteration
        
        # Calculate J_best(t) up to the current iteration
        j_best = calculate_j_best(distortions)

        # Prepare for the next iteration
        kmeans.max_iter += 1

    return kmeans, calculate_j_best(distortions)

# Function to calculate J_best(t)
def calculate_j_best(distortions):
    """
    Calculates J_best(t) by taking the cumulative minimum of the distortions.
    """
    distortions = np.array(distortions)
    j_best = np.minimum.accumulate(distortions)
    return j_best
