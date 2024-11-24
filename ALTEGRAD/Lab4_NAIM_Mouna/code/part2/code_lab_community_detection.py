"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
     #Compute the adjacency matrix of the graph
    adjacency_matrix = nx.adjacency_matrix(G).toarray()

    # Compute the degree matrix D
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

    # Compute I - inverse(D)*A
    laplacian_matrix = np.linalg.inv(degree_matrix) @ adjacency_matrix
    laplacian_matrix = np.identity(adjacency_matrix.shape[0]) - laplacian_matrix

    # Perform eigenvalue decomposition to get the first k eigenvectors
    eigvals, eigvecs = np.linalg.eigh(laplacian_matrix)
    sorted_indices = np.argsort(eigvals)
    selected_eigvecs = eigvecs[:, sorted_indices[:k]]

    # Apply k-means to rows of selected eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(selected_eigvecs)
    labels = kmeans.labels_

    # Assign each node to a cluster
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}
    ##################
    
    
    
    return clustering


############## Task 4

##################
__path__ = '../datasets/CA-HepTh.txt'
G = nx.read_edgelist(__path__, comments = '#', delimiter='\t') #Load the data into the graph G
gcc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(gcc_nodes)
k = 50
clustering = spectral_clustering(G, k)
#print(clustering)
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    m = G.number_of_edges()
    
    # Initialize modularity
    modularity = 0
    
    # Group nodes by clusters
    clusters = {}
    for node, cluster_id in clustering.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    # Compute modularity
    for cluster_id, nodes in clusters.items():
        # Induced subgraph for the cluster
        subgraph = G.subgraph(nodes)
        
        # Number of edges within the cluster
        lc = subgraph.number_of_edges()
        
        # Sum of degrees of nodes in the cluster
        dc = sum(dict(G.degree(nodes)).values())
        
        # Update modularity
        modularity += (lc / m) - (dc / (2 * m)) ** 2
    ##################
    
    
    return modularity



############## Task 6

##################
# Modularity for the clusters obtained by Spectral Clustering algorithm using k=50
print(modularity(gcc, clustering))

#For random partition
random_clustering = {node: randint(0, 49) for node in gcc.nodes()}
print(modularity(gcc, random_clustering))
##################







