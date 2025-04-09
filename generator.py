"""
This module sends the query and retrieved context to the Gemini 2.0 Flash model
to generate a human-like answer.

It uses the Google Generative AI Python client.

Steps:
- Format the final prompt with query + context.
- Send it to Gemini.
- Return the generated response.
"""

import os
from dotenv import load_dotenv 
from google.generativeai import configure, GenerativeModel

load_dotenv()

configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = GenerativeModel("gemini-2.0-flash")

def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text
