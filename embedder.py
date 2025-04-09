"""
This module is responsible for converting chunks of text into embeddings (vectors).
It uses a pre-trained SentenceTransformer model to generate these vectors.

Embeddings allow us to compare the similarity between a user query and
stored text, even if they don't use the exact same words.

Usage:
- Load model once and call embed_text() with a list of strings to get their embeddings.
"""

from sentence_transformers import SentenceTransformer

# Load a compact model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts):
    return model.encode(texts).tolist()
