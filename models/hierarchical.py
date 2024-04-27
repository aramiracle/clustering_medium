from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_hierarchical(embeddings, true_labels, n_clusters):
    """Cluster embeddings using Hierarchical Clustering."""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    predicted_labels = hierarchical.fit_predict(embeddings)
    accuracy = accuracy_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, accuracy, silhouette, adjusted_rand, nmi