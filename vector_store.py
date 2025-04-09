"""
This module handles storage and retrieval of embeddings using FAISS,
a fast similarity search library.

We use it to:
- Save and load the FAISS index of vectorized documents.
- Add new embeddings.
- Search for the most similar documents to a given query vector.

It also keeps a mapping from index position to the original text chunk.
"""

import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.text_chunks = []

    def build_index(self, embeddings, texts):
        self.text_chunks = texts
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_embedding, k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return [self.text_chunks[i] for i in I[0]]
