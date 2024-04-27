import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, pairwise_distances

def grid_search_dbscan(embeddings, true_labels):
    """Grid search for DBSCAN parameters."""
    print("Begin clustering embeddings with DBSCAN...")
    
    # Compute pairwise distances
    distances = pairwise_distances(embeddings)
    min_dist = np.min(distances[np.nonzero(distances)])  # Minimum non-zero distance
    max_dist = np.max(distances)  # Maximum distance
    
    # Update parameter grid
    param_grid = {
        'eps': np.linspace(np.log10(min_dist), np.log10(max_dist), 20),
        'min_samples': np.arange(1, 11)
    }
    
    best_score = -1
    best_params = None
    for params in tqdm(ParameterGrid(param_grid), desc="Grid search"):
        eps = params['eps']
        min_samples = params['min_samples']
        _, _, score, _, _ = cluster_embeddings_dbscan(embeddings, true_labels, eps, min_samples)
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params

def cluster_embeddings_dbscan(embeddings, true_labels, eps, min_samples):
    """Cluster embeddings using DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predicted_labels = dbscan.fit_predict(embeddings)
    if len(np.unique(predicted_labels)) > 1:  # Check if there's more than one cluster
        accuracy = accuracy_score(true_labels, predicted_labels)
        silhouette = silhouette_score(embeddings, predicted_labels)
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    else:
        # If only one cluster, set scores to 0
        accuracy, silhouette, adjusted_rand, nmi = 0, 0, 0, 0
    return predicted_labels, accuracy, silhouette, adjusted_rand, nmi