from transformers import MobileBertModel, MobileBertTokenizer
from tqdm import tqdm
import torch
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

categories = [
        'alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space',
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'
    ]

def load_mobilebert_model_and_tokenizer(model_name='google/mobilebert-uncased'):
    """Load pre-trained MobileBERT model and tokenizer."""
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    model = MobileBertModel.from_pretrained(model_name)
    return model, tokenizer

def encode_single_article(inputs_tokenizer):
    index, article, tokenizer = inputs_tokenizer
    with torch.no_grad():
        inputs = tokenizer.encode_plus(article, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    return index, inputs

def encode_articles(encoded_articles, tokenizer):
    """Encode the articles using the provided tokenizer."""
    num_articles = len(encoded_articles)

    with ThreadPoolExecutor() as executor:
        encoded_with_index = [(index, article, tokenizer) for index, article in enumerate(encoded_articles)]
        encoded_inputs = [inputs for _, inputs in tqdm(executor.map(encode_single_article, encoded_with_index),
                                     total=num_articles,
                                     desc="Encoding articles")]

    return encoded_inputs

def extract_single_embedding(inputs_model):
    index, inputs, model = inputs_model
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state.squeeze(), dim=0).numpy()
    return index, embedding

def extract_and_concatenate_embeddings(encoded_articles, model, embedding_dim=512):
    """Extract and concatenate embeddings from the encoded articles using the provided model."""
    num_articles = len(encoded_articles)

    # Create an array of zeros to hold the embeddings
    concatenated_embeddings = np.zeros((num_articles, embedding_dim))

    with ThreadPoolExecutor() as executor:
        embeddings_with_index = [(index, inputs, model) for index, inputs in enumerate(encoded_articles)]
        for index, embedding in tqdm(executor.map(extract_single_embedding, embeddings_with_index),
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
    
def print_cluster_metrics(method_name, accuracy, silhouette, adjusted_rand, nmi):
    """Print clustering metrics."""
    print(f"Method: {method_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Silhouette Score: {silhouette}")
    print(f"Adjusted Rand Score: {adjusted_rand}")
    print(f"Normalized Mutual Information (NMI): {nmi}\n")