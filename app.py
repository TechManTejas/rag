"""
A simple Streamlit interface to use the RAG system.

Steps:
- Takes user input from a text field.
- Passes it through the RAG pipeline.
- Displays the generated answer in the browser.

Ideal for testing and quick prototyping using a web interface.
"""

import streamlit as st
from rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("💬 RAG System Demo")

st.write("Ask a question below and get an AI-generated answer using the RAG pipeline.")

query = st.text_input("Enter your question:", placeholder="e.g. What is Retrieval-Augmented Generation?")

if query:
    with st.spinner("Thinking..."):
        answer = run_rag_pipeline(query)
    st.markdown("### Answer")
    st.write(answer)
