"""
A simple Streamlit interface to use the RAG system.

Steps:
- Allows user to upload a .txt file or use default data
- Takes user input from a text field
- Passes it through the RAG pipeline
- Displays the generated answer in the browser

Ideal for testing and quick prototyping using a web interface.
"""

import streamlit as st
import os
import tempfile
from rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("💬 RAG System Demo")

st.write("Upload your own text file or use the default data, then ask a question.")

# File upload widget
uploaded_file = st.file_uploader("Upload a .txt file (optional)", type="txt")

# Initialize file_path variable
file_path = 'data.txt'  # Default file path

# Process uploaded file if present
if uploaded_file is not None:
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        # Write the uploaded file content to the temp file
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    
    st.success(f"File uploaded successfully!")
    
    # Show a preview of the uploaded content
    file_content = uploaded_file.getvalue().decode('utf-8')
    with st.expander("Preview uploaded content"):
        st.text(file_content[:500] + "..." if len(file_content) > 500 else file_content)
else:
    st.info("Using default data.txt file")

# Query input
query = st.text_input("Enter your question:", placeholder="e.g. What is Retrieval-Augmented Generation?")

if query:
    with st.spinner("Thinking..."):
        answer = run_rag_pipeline(query, file_path)
    st.markdown("### Answer")
    st.write(answer)
    
    # Clean up temporary file if one was created
    if uploaded_file is not None and file_path != 'data.txt':
        try:
            os.unlink(file_path)
        except:
            pass 