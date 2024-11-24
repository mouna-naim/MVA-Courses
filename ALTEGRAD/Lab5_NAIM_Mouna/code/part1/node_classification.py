"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
##################
nx.draw_networkx(G, with_labels=True, node_color=y, cmap=plt.cm.rainbow)
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim) # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# Train a Logistic Regression model
logistic_reg = LogisticRegression(random_state=42)
logistic_reg.fit(X_train, y_train)

# Make predictions
y_pred = logistic_reg.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy of Logistic Regression: {accuracy:.2f}%")
##################


############## Task 8
# Generates spectral embeddings

##################
laplacian_matrix = nx.laplacian_matrix(G).astype(float)
eigenvalues, eigenvectors = eigs(laplacian_matrix, k=2, which="SR")
laplacian_embeddings = eigenvectors.real

# Split embeddings into training and testing data
X_train = laplacian_embeddings[idx_train, :]
X_test = laplacian_embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]

# Train a Logistic Regression model
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train, y_train)

# Make predictions 
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy of Logistic Regression on Laplacian Embedding: {accuracy:.2f}%")
##################
