"""
This module formats the final prompt that is sent to the language model.

It combines:
- The original user query.
- The retrieved context chunks.

The prompt is designed to help the AI answer the question using only the provided information,
making the system grounded and trustworthy.
"""

def build_prompt(query, contexts):
    context_str = "\n\n".join(contexts)
    return f"""Use the following information to answer the question:

{context_str}

Question: {query}
"""
