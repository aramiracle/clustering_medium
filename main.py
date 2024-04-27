from sklearn.datasets import fetch_20newsgroups
from utils import categories, load_mobilebert_model_and_tokenizer, load_concatenated_embeddings, encode_articles, extract_and_concatenate_embeddings, save_concatenated_embeddings, print_cluster_metrics

from models.kmeans import cluster_embeddings_kmeans
from models.hierarchical import cluster_embeddings_hierarchical
from models.dbscan import grid_search_dbscan, cluster_embeddings_dbscan
from models.mean_shift import cluster_embeddings_mean_shift
from models.gmm import cluster_embeddings_gmm
from models.spectral import cluster_embeddings_spectral

def main():
    # Load MobileBERT model and tokenizer
    model, tokenizer = load_mobilebert_model_and_tokenizer()

    # Fetch the 20 Newsgroups dataset
    newsgroups_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

    # Check if concatenated embeddings are already saved
    saved_embeddings = load_concatenated_embeddings()
    if saved_embeddings is None:
        # Encode articles
        encoded_articles = encode_articles(newsgroups_data.data, tokenizer)

        # Extract and concatenate embeddings
        concatenated_embeddings = extract_and_concatenate_embeddings(encoded_articles, model)

        # Save concatenated embeddings
        save_concatenated_embeddings(concatenated_embeddings)
    else:
        concatenated_embeddings = saved_embeddings

    # Cluster embeddings using KMeans
    kmeans_results = cluster_embeddings_kmeans(concatenated_embeddings, newsgroups_data.target, n_clusters=20)
    print_cluster_metrics("KMeans", *kmeans_results)

    # Cluster embeddings using Hierarchical Clustering
    hierarchical_results = cluster_embeddings_hierarchical(concatenated_embeddings, newsgroups_data.target, n_clusters=20)
    print_cluster_metrics("Hierarchical Clustering", *hierarchical_results)

    # Grid search for best DBSCAN parameters
    best_params = grid_search_dbscan(concatenated_embeddings, newsgroups_data.target)
    print("Best DBSCAN parameters:", best_params)

    # Cluster embeddings using DBSCAN with best parameters
    dbscan_results = cluster_embeddings_dbscan(concatenated_embeddings, newsgroups_data.target, eps=best_params['eps'], min_samples=best_params['min_samples'])
    print_cluster_metrics("DBSCAN", *dbscan_results)

    # Cluster embeddings using Mean Shift Clustering
    mean_shift_results = cluster_embeddings_mean_shift(concatenated_embeddings, newsgroups_data.target)
    print_cluster_metrics("Mean Shift Clustering", *mean_shift_results)

    # Cluster embeddings using Gaussian Mixture Models (GMM)
    gmm_results = cluster_embeddings_gmm(concatenated_embeddings, newsgroups_data.target, n_components=20)
    print_cluster_metrics("Gaussian Mixture Models", *gmm_results)

    # Cluster embeddings using Spectral Clustering
    spectral_results = cluster_embeddings_spectral(concatenated_embeddings, newsgroups_data.target, n_clusters=20)
    print_cluster_metrics("Spectral Clustering", *spectral_results)

if __name__ == "__main__":
    main()
