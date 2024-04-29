from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score, completeness_score

def cluster_embeddings_mini_batch_kmeans(embeddings, true_labels, n_clusters):
    """Cluster embeddings using Mini Batch KMeans."""
    print("Begin clustering embeddings with Mini Batch KMeans...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=20, random_state=24)
    predicted_labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, predicted_labels)
    fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    c = completeness_score(true_labels, predicted_labels)
    return predicted_labels, silhouette, fowlkes_mallows, adjusted_rand, nmi, c