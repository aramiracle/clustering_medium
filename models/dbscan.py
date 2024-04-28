import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, pairwise_distances

def cluster_embeddings_dbscan(embeddings, true_labels, eps, min_samples):
    """Cluster embeddings using DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predicted_labels = dbscan.fit_predict(embeddings)
    if len(np.unique(predicted_labels)) > 1:  # Check if there's more than one cluster
        fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
        silhouette = silhouette_score(embeddings, predicted_labels)
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    else:
        # If only one cluster, set scores to 0
        fowlkes_mallows, silhouette, adjusted_rand, nmi = 0, 0, 0, 0
    return predicted_labels, fowlkes_mallows, silhouette, adjusted_rand, nmi

def custom_scorer(estimator, X, true_labels):
    """Custom scorer based on Fowlkes-Mallows score for DBSCAN."""
    predicted_labels = estimator.fit_predict(X)
    score = fowlkes_mallows_score(true_labels, predicted_labels)
    return -score

def grid_search_dbscan(embeddings, true_labels):
    """Grid search for DBSCAN parameters."""
    print("Begin clustering embeddings with DBSCAN...")
    
    # Compute pairwise distances
    distances = pairwise_distances(embeddings)
    min_dist = np.min(distances[np.nonzero(distances)])  # Minimum non-zero distance
    max_dist = np.max(distances)  # Maximum distance
    
    # Update parameter grid
    param_grid = {
        'eps': np.logspace(np.log10(min_dist), np.log10(max_dist), 30),
        'min_samples': [1, 3, 5, 10, 20]
    }

    # Instantiate DBSCAN object
    dbscan = DBSCAN()

    # Instantiate GridSearchCV with custom scorer
    print("Begin grid search for best clustering...")
    grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, scoring=custom_scorer)

    # Fit GridSearchCV to data
    grid_search.fit(embeddings, true_labels)

    # Retrieve best parameters and best score
    best_params = grid_search.best_params_

    return best_params