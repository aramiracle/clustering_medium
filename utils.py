from tqdm import tqdm
import torch
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

def extract_single_embedding(articles_with_index):
    index, article = articles_with_index
    # Load feature extractor model
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
    model.max_seq_length = 1024

    with torch.no_grad():
        embedding = model.encode([article], convert_to_numpy=True)

    return index, embedding

def extract_and_concatenate_embeddings(articles, embedding_dim=768):
    """Extract and concatenate embeddings from the encoded articles using the provided model."""
    num_articles = len(articles)

    # Create an array of zeros to hold the embeddings
    concatenated_embeddings = np.zeros((num_articles, embedding_dim))

    with ThreadPoolExecutor(max_workers=4) as executor:
        articles_with_index = [(index, article) for index, article in enumerate(articles)]
        for index, embedding in tqdm(executor.map(extract_single_embedding, articles_with_index),
                                     total=num_articles,
                                     desc="Extracting embeddings"):
            # Update the corresponding row in the embeddings array
            concatenated_embeddings[index] = embedding

    return concatenated_embeddings

def save_concatenated_embeddings(embeddings, directory='features'):
    """Save concatenated embeddings."""
    os.makedirs(directory, exist_ok=True)
    
    np.save(os.path.join(directory, 'concatenated_embeddings.npy'), embeddings)

def load_concatenated_embeddings(directory='features'):
    """Load concatenated embeddings if they exist."""
    embeddings_file = os.path.join(directory, 'concatenated_embeddings.npy')
    if os.path.exists(embeddings_file):
        print("Loading embeddings...")
        return np.load(embeddings_file)
    else:
        return None
    
def print_cluster_metrics(method_name, fowlkes_mallows, silhouette, adjusted_rand, nmi):
    """Print clustering metrics."""
    print(f"Method: {method_name}")
    print(f"Fowlkes Mallows Score: {round(fowlkes_mallows, 4)}")
    print(f"Silhouette Score: {round(silhouette, 4)}")
    print(f"Adjusted Rand Score: {round(adjusted_rand, 4)}")
    print(f"Normalized Mutual Information (NMI): {round(nmi, 4)}\n")