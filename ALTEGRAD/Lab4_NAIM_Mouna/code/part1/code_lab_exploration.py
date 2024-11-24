"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#from pathlib import Path

############## Task 1

__path__ = '../datasets/CA-HepTh.txt'
G = nx.read_edgelist(__path__, comments = '#', delimiter='\t') #Load the data into the graph G
nodes_G = G.number_of_nodes() #Extract the number of nodes 
edges_G = G.number_of_edges() #Extract the number of edges
print(f"Total number of edges in G: {edges_G}")
print(f"Total number of nodes in G: {nodes_G}")


############## Task 2

# The number of connected components
num_connected_components = nx.number_connected_components(G)
print(f"Number of connected components: {num_connected_components}")

# The largest connected component
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_subgraph = G.subgraph(largest_cc)

# Compute the number of nodes and edges in the largest connected component
num_nodes_largest_cc = largest_cc_subgraph.number_of_nodes()
num_edges_largest_cc = largest_cc_subgraph.number_of_edges()

# Compute fractions of nodes and edges in the largest connected component
fraction_nodes = num_nodes_largest_cc / nodes_G
fraction_edges = num_edges_largest_cc / edges_G

print(f"Number of nodes in the largest connected component: {num_nodes_largest_cc}")
print(f"Number of edges in the largest connected component: {num_edges_largest_cc}")
print(f"Fraction of nodes in the largest connected component: {fraction_nodes:.2f}")
print(f"Fraction of edges in the largest connected component: {fraction_edges:.2f}")


