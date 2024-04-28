from sklearn.cluster import Birch
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_birch(embeddings, true_labels, n_clusters):
    """Cluster embeddings using Birch."""
    print("Begin clustering embeddings with Birch...")
    hierarchical = Birch(n_clusters=n_clusters)
    predicted_labels = hierarchical.fit_predict(embeddings)
    fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, fowlkes_mallows, silhouette, adjusted_rand, nmi