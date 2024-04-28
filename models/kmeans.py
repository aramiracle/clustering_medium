from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_kmeans(embeddings, true_labels, n_clusters):
    """Cluster embeddings using KMeans."""
    print("Begin clustering embeddings with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)
    fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, fowlkes_mallows, silhouette, adjusted_rand, nmi