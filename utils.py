from transformers import MobileBertModel, MobileBertTokenizer
from tqdm import tqdm
import torch
import numpy as np
import os

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

def encode_articles(articles, tokenizer, max_length=512):
    """Encode articles using the provided tokenizer."""
    encoded_articles = []
    for article in tqdm(articles, desc="Encoding articles"):
        inputs = tokenizer.encode_plus(article, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        encoded_articles.append(inputs)
    return encoded_articles

def extract_and_concatenate_embeddings(encoded_articles, model):
    """Extract and concatenate embeddings from the encoded articles using the provided model."""
    concatenated_embeddings = None
    for inputs in tqdm(encoded_articles, desc="Extracting embeddings"):
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.squeeze().numpy()
        if concatenated_embeddings is None:
            concatenated_embeddings = embedding
        else:
            concatenated_embeddings = np.concatenate((concatenated_embeddings, embedding), axis=0)
    return concatenated_embeddings

def save_concatenated_embeddings(embeddings, directory='features'):
    """Save concatenated embeddings."""
    os.makedirs(directory, exist_ok=True)
    
    np.save(os.path.join(directory, 'concatenated_embeddings.npy'), embeddings)

def load_concatenated_embeddings(directory='features'):
    """Load concatenated embeddings if they exist."""
    embeddings_file = os.path.join(directory, 'concatenated_embeddings.npy')
    if os.path.exists(embeddings_file):
        return np.load(embeddings_file)
    else:
        return None
    
def print_cluster_metrics(method_name, accuracy, silhouette, adjusted_rand, nmi):
    """Print clustering metrics."""
    print(f"Method: {method_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Silhouette Score: {silhouette}")
    print(f"Adjusted Rand Score: {adjusted_rand}")
    print(f"Normalized Mutual Information (NMI): {nmi}")