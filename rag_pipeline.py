"""
This is the core pipeline that connects all parts of the RAG system.

Steps:
1. Load and embed the text from data.txt (if not already embedded).
2. Store the vectors in FAISS.
3. On user query:
   - Embed the query.
   - Retrieve top-k relevant chunks.
   - Format the prompt.
   - Generate the final answer using Gemini.

Can be imported into other scripts or used directly.
"""
from retriever import Retriever
from generator import generate_answer
from prompt import build_prompt

def read_text_chunks(file_path, chunk_size=300):
    with open(file_path, 'r') as f:
        words = f.read().split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = read_text_chunks("data.txt")
retriever = Retriever(chunks)

def run_rag_pipeline(query):
    top_chunks = retriever.retrieve(query)
    prompt = build_prompt(query, top_chunks)
    return generate_answer(prompt)