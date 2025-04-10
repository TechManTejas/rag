"""
This module is used to find the most relevant chunks of text given a user query.

It performs the following:
- Converts the user query to an embedding using the same embedder.
- Searches the FAISS vector store for the top-k most similar chunks.

Returns a list of context strings that will be used by the generator.
"""

from embedder import embed_text
from vector_store import VectorStore

class Retriever:
    _instance = None

    def __new__(cls, chunks):
        if cls._instance is None:
            cls._instance = super(Retriever, cls).__new__(cls)
            cls._instance.vs = VectorStore()
            embeddings = embed_text(chunks)
            cls._instance.vs.build_index(embeddings, chunks)
        return cls._instance

    def retrieve(self, query, k=3):
        query_embedding = embed_text([query])[0]
        return self.vs.search(query_embedding, k)