from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score, completeness_score

def cluster_embeddings_gmm(embeddings, true_labels, n_components):
    """Cluster embeddings using Gaussian Mixture Models."""
    print("Begin clustering embeddings with Gaussian Mixture Models...")
    gmm = GaussianMixture(n_components=n_components, random_state=24)
    predicted_labels = gmm.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, predicted_labels)
    fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    c = completeness_score(true_labels, predicted_labels)
    return predicted_labels, silhouette, fowlkes_mallows, adjusted_rand, nmi, c