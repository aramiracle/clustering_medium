from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_embeddings_gmm(embeddings, true_labels, n_components):
    """Cluster embeddings using Gaussian Mixture Models."""
    print("Begin clustering embeddings with Gaussian Mixture Models...")
    gmm = GaussianMixture(n_components=n_components)
    predicted_labels = gmm.fit_predict(embeddings)
    accuracy = accuracy_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return predicted_labels, accuracy, silhouette, adjusted_rand, nmi