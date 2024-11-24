"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    """
    G: The graph
    node: Starting node for the rundom walk 
    walk_length: Length of the random walk

    Returns a list of nodes representing the random walk
    """
    ##################
    walk = [node]  # Initialize the walk with the starting node

    for _ in range(walk_length - 1):  # Perform the walk
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) > 0:  # Check if the current node has neighbors
            next_node = np.random.choice(neighbors)  # Randomly choose the next node
            walk.append(next_node)
        else:
            break  # End the walk if the node has no neighbors
    ##################
    
    walk = [str(node) for node in walk]
        
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    # Iterate over the number of walks
    for _ in range(num_walks):
        nodes = list(G.nodes())  # Get all nodes in the graph
        np.random.shuffle(nodes)  # Introduce randomness
        
        # Perform a random walk for each node
        for node in nodes:
            walk = random_walk(G, node, walk_length)  # Call the random_walk function
            walks.append(walk)
    
    # Shuffle the generated walks to ensure randomness
    permuted_walks = np.random.permutation(walks)    
    ##################

    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
