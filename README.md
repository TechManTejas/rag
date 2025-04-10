
# Retrieval-Augmented Generation (RAG) System

## 📌 Overview

This project implements a minimal RAG (Retrieval-Augmented Generation) pipeline using:

-   **Streamlit** for the user interface
    
-   **SentenceTransformers** for generating embeddings
    
-   **FAISS** for similarity search
    
-   **Gemini** (or another LLM) for final answer generation
    

## 🔁 System Workflow (High-Level)

```text
              ┌─────────────────────────────────────────────────────┐
              │                  SYSTEM WORKFLOW                    │
              └─────────────────────────────────────────────────────┘

                          User Inputs Text Chunks (Documents)
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │ SentenceTransformer Model   │
                         │  (Embeds each chunk)        │
                         └─────────────────────────────┘
                                       │
                          Produces Embeddings (Vectors)
                                       │
                                       ▼
                            ┌────────────────────┐
                            │   FAISS Indexing   │
                            │  (Vector Database) │
                            └────────────────────┘
                                       
                                       
                        ┌─────────────────────────────┐
                        │  User Query (e.g. question) │
                        └─────────────────────────────┘
                                       │
                          Query is embedded (vectorized)
                                       │
                                       ▼
                       ┌──────────────────────────────────┐
                       │ Vector Similarity Search (FAISS) │
                       └──────────────────────────────────┘
                                       │
                      Returns Top-K Most Relevant Chunks
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │ Language Model (e.g. Gemini)│
                         │ Generates Final Response    │
                         └─────────────────────────────┘

```

----------

## 🧠 Example Walkthrough

### 1. Input Chunks

```python
chunks: List[str] = [
    "RAG stands for Retrieval-Augmented Generation.",
    "It improves language model accuracy by grounding answers in facts.",
    "We use FAISS to search relevant chunks.",
    "Gemini generates the final answer."
]

```

### 2. Generate Embeddings

Each chunk is converted to a 384-dimensional vector.

```python
embeddings: List[List[float]] = [
    [0.12, -0.03, ..., 0.45],  # Embedding for chunk 1
    [0.11, -0.05, ..., 0.42],
    [0.08, -0.06, ..., 0.33],
    [0.10, -0.02, ..., 0.29],
]

```

### 3. Store in FAISS

FAISS index stores vectors for fast similarity search.

```text
FAISS Index (L2 Distance)
┌────┬────────────────────────────┐
│ ID │ Vector                     │
├────┼────────────────────────────┤
│ 0  │ [0.12, -0.03, ..., 0.45]   │
│ 1  │ [0.11, -0.05, ..., 0.42]   │
│ 2  │ [0.08, -0.06, ..., 0.33]   │
│ 3  │ [0.10, -0.02, ..., 0.29]   │
└────┴────────────────────────────┘

```

### 4. Query Time!

```python
query = "What is RAG?"
query_embedding = embed_text([query])[0]  # [0.13, -0.04, ..., 0.44]

```

### 5. Search Top-K

```text
I = [[0, 1, 2]]  # Indices of top-3 closest vectors

```

### 6. Final Retrieved Chunks

```python
[
    "RAG stands for Retrieval-Augmented Generation.",
    "It improves language model accuracy by grounding answers in facts.",
    "We use FAISS to search relevant chunks."
]

```

## ✅ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

```
    

## ❤️ Credits

-   [SentenceTransformers](https://www.sbert.net/)
    
-   [FAISS](https://github.com/facebookresearch/faiss)
    
-   [Streamlit](https://streamlit.io/)