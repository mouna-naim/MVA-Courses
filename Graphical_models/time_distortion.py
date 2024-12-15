
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm  
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import seaborn as sns  
from sklearn.cluster import KMeans

def evaluate_clustering_methods_over_runs(X, true_labels, n_clusters, max_iter, n_runs, clustering_methods):
    """
    Evaluates multiple clustering methods by calculating the final distortions (J_best) over multiple runs.
    Calculates accuracy and displays the confusion matrix for each method.

    Args:

    X: ndarray, input data.
    true_labels: ndarray, true labels of the data.
    n_clusters: int, number of clusters.
    max_iter: int, maximum number of iterations for clustering.
    n_runs: int, number of independent runs.
    clustering_methods: dict, dictionary containing the names and clustering functions {name: function}.
    
    """
    results = {}
    plt.figure(figsize=(10, 6))
    
    # List of probabilistic methods
    probabilistic_methods = ["KMeans++", "PCA-guided KMeans", "k-means R1", "k-means R2", "PCA+kmeans"]
    deterministic_colors = ['red', 'green', 'purple', 'orange']  # Colors for deterministic methods
    deterministic_styles = ['--', '-.', ':', '-']  
    deterministic_counter = 0  
    
    # Parameters for stagnation detection
    stagnation_threshold = 3  # Number of consecutive runs required to declare stagnation
    epsilon = 1e-4  # Threshold of variation for stagnation detection

    
    for method_name, clustering_function in clustering_methods.items():
        print(f"Evaluating method: {method_name}")
        if method_name in probabilistic_methods: 
            distortions = []
            accuracies = []
            best_stagnation_model = None
            stagnation_iteration = None
            previous_j_best = None  
            stagnation_counter = 0  
            start_time = time.time()
            
            # Loop over the runs with a progress bar
            for run in tqdm(range(n_runs), desc=f"{method_name}", leave=True, ncols=100):
                kmeans_model, j_best = clustering_function(X, n_clusters, max_iter, run)
                final_j_best = j_best[-1]
                distortions.append(final_j_best)
                
                # Calculation of the variation compared to the previous run
                if previous_j_best is not None:
                    variation = abs(final_j_best - previous_j_best)
                    if variation < epsilon:
                        stagnation_counter += 1
                        if stagnation_counter >= stagnation_threshold and best_stagnation_model is None:
                            best_stagnation_model = kmeans_model
                            stagnation_iteration = run + 1  # Conversion of the index to 1-based
                    else:
                        stagnation_counter = 0 
                previous_j_best = final_j_best  
                
                # Accuracy calculation
                predicted_labels = kmeans_model.labels_
                accuracy = calculate_clustering_accuracy(true_labels, predicted_labels, n_clusters)
                accuracies.append(accuracy)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Sorting distortions in descending order
            distortions_sorted = sorted(distortions, reverse=True)
            results[method_name] = {
                "distortions": distortions_sorted,
                "execution_time": execution_time,
                "accuracy": np.mean(accuracies),
                "kmeans_best": best_stagnation_model,
                "stagnation_iteration": stagnation_iteration,
                "accuracy_best": calculate_clustering_accuracy(true_labels, best_stagnation_model.labels_, n_clusters) if best_stagnation_model is not None else None
            }
            
            # Plot of sorted distortions
            runs = range(1, n_runs + 1)
            plt.plot(runs, distortions_sorted, label=f"{method_name}")
        
        else:  # Deterministic methods
            start_time = time.time()
            
            # Loop over a single run with a progress bar
            for _ in tqdm(range(1), desc=f"{method_name}", leave=True, ncols=100):
                kmeans_model, j_best = clustering_function(X, n_clusters, max_iter, 0)
                final_j_best = j_best[-1]
                
                # Accuracy calculation
                predicted_labels = kmeans_model.labels_
                accuracy = calculate_clustering_accuracy(true_labels, predicted_labels, n_clusters)
            
            end_time = time.time()
            execution_time = end_time - start_time
            results[method_name] = {
                "distortion": final_j_best,
                "execution_time": execution_time,
                "accuracy": accuracy,
                "kmeans_best": kmeans_model,
                "stagnation_iteration": None
            }
            
            # Plot of a horizontal line for deterministic methods
            color = deterministic_colors[deterministic_counter % len(deterministic_colors)]
            style = deterministic_styles[deterministic_counter % len(deterministic_styles)]
            plt.axhline(y=final_j_best, color=color, linestyle=style, label=f"{method_name}")
            deterministic_counter += 1
    
    # Plot configuration
    plt.xlabel("Runs")
    plt.ylabel("Final J_best")
    plt.title("Final J_best over multiple runs for different clustering methods")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    plt.grid(alpha=0.5)
    plt.show()
    
    # Display confusion matrices after the plot of distortions
    for method_name, clustering_function in clustering_methods.items():
        print(f"Confusion matrix for {method_name}:")
        if method_name in probabilistic_methods:
            # Display the confusion matrix for the model at stagnation or at the last run
            kmeans_model = results[method_name]["kmeans_best"] if results[method_name]["kmeans_best"] is not None else clustering_function(X, n_clusters, max_iter, n_runs - 1)[0]
            predicted_labels = kmeans_model.labels_
        else:
            # Display the confusion matrix for deterministic methods
            kmeans_model = clustering_function(X, n_clusters, max_iter, 0)[0]
            predicted_labels = kmeans_model.labels_
        
        # Calculate the confusion matrix with reassignment
        conf_matrix = calculate_confusion_matrix_with_reassignment(true_labels, predicted_labels, n_clusters)
        
        # Determine the optimal figure size based on the number of clusters
        fig_width = max(10, n_clusters * 0.4)
        fig_height = max(6, n_clusters * 0.3)
        
        # Plot the confusion matrix
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xticks(ticks=np.arange(n_clusters) + 0.5, labels=np.arange(n_clusters), rotation=90, fontsize=10, ha='center')
        plt.yticks(ticks=np.arange(n_clusters) + 0.5, labels=np.arange(n_clusters), rotation=0, fontsize=10, va='center')
        plt.title(f"Confusion Matrix for {method_name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.show()

    # Create the summary table of execution times and accuracies
    execution_times = []
    for method_name, result in results.items():
        execution_time = result.get("execution_time", 0) if result.get("execution_time") is not None else 0
        accuracy_best = result.get("accuracy_best", result["accuracy"]) if result.get("accuracy_best") is not None else result["accuracy"]
        accuracy_best = 0 if accuracy_best is None else accuracy_best

        # Format the results
        execution_times.append({
            "Method": method_name, 
            "Execution Time (s)": f"{execution_time:.6f}", 
            "Accuracy": f"{accuracy_best:.6f}"
        })

    execution_times_df = pd.DataFrame(execution_times).sort_values(by="Accuracy", ascending=False)
    print("\nSummary table of execution times and accuracies:")
    print(tabulate(execution_times_df, headers="keys", tablefmt="pretty", floatfmt=".6f"))

def calculate_clustering_accuracy(true_labels, predicted_labels, n_clusters):
    """
    Calculates the accuracy of clustering using the Hungarian algorithm to align predicted labels with true labels.

    Args:
    - true_labels : ndarray, true labels.
    - predicted_labels : ndarray, labels predicted by the clustering algorithm.
    - n_clusters : int, number of clusters.

    Returns:
    - accuracy : float, clustering accuracy.
    """
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], predicted_labels[i]] += 1
    
    # Use the Hungarian algorithm to find the best alignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    total_correct = contingency_matrix[row_ind, col_ind].sum()
    accuracy = total_correct / len(true_labels)
    
    return accuracy

def calculate_confusion_matrix_with_reassignment(true_labels, predicted_labels, n_clusters):
    """
    Recalculates the confusion matrix after optimally reassigning cluster labels to true labels.

    Args:
    - true_labels : ndarray, true labels.
    - predicted_labels : ndarray, labels predicted by the clustering algorithm.
    - n_clusters : int, number of clusters.

    Returns:
    - conf_matrix : ndarray, confusion matrix after reassignment.
    """
    # Build the contingency matrix
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], predicted_labels[i]] += 1
    
    # Find the optimal assignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Create a mapping of predicted labels to true labels
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    reassigned_labels = np.array([label_mapping[label] for label in predicted_labels])
    
    # Calculate the confusion matrix after reassignment
    conf_matrix = confusion_matrix(true_labels, reassigned_labels)
    
    return conf_matrix
