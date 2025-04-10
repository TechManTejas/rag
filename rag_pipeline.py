"""
This is the core pipeline that connects all parts of the RAG system.

Steps:
1. Load and embed the text from the specified file (if not already embedded).
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    except Exception as e:
        # Return an error message as a chunk if file can't be read
        return [f"Error reading file: {str(e)}"]

def run_rag_pipeline(query, file_path='data.txt'):
    chunks = read_text_chunks(file_path)
    
    # Check if we have valid chunks
    if not chunks or (len(chunks) == 1 and chunks[0].startswith("Error")):
        return "I couldn't process the file. Please make sure it's a valid text file and try again."
    
    retriever = Retriever(chunks)
    top_chunks = retriever.retrieve(query)
    prompt = build_prompt(query, top_chunks)
    return generate_answer(prompt)