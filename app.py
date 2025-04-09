"""
A simple command-line interface (CLI) to use the RAG system.

Steps:
- Takes user input from the terminal.
- Passes it through the RAG pipeline.
- Prints the generated answer to the screen.

Ideal for testing and quick prototyping.
"""

from rag_pipeline import run_rag_pipeline

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = run_rag_pipeline(query)
        print("\n", answer)
