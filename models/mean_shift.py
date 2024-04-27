from sklearn.cluster import MeanShift
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_mean_shift(embeddings, true_labels):
    """Cluster embeddings using Mean Shift Clustering."""
    ms = MeanShift()
    predicted_labels = ms.fit_predict(embeddings)
    accuracy = accuracy_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, accuracy, silhouette, adjusted_rand, nmi