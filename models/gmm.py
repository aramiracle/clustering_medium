from sklearn.mixture import GaussianMixture
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_gmm(embeddings, true_labels, n_components):
    """Cluster embeddings using Gaussian Mixture Models."""
    print("Begin clustering embeddings with Gaussian Mixture Models...")
    gmm = GaussianMixture(n_components=n_components, random_state=24)
    predicted_labels = gmm.fit_predict(embeddings)
    fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, fowlkes_mallows, silhouette, adjusted_rand, nmi