import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def grid_search_dbscan(embeddings, true_labels):
    """Grid search for DBSCAN parameters."""
    param_grid = {
        'eps': np.linspace(0.1, 2.0, 20),
        'min_samples': np.arange(1, 10)
    }
    best_score = -1
    best_params = None
    for params in tqdm(ParameterGrid(param_grid), desc="Grid search"):
        eps = params['eps']
        min_samples = params['min_samples']
        _, _, _, _, score = cluster_embeddings_dbscan(embeddings, true_labels, eps, min_samples)
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