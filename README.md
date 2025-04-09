# RAG

A minimalist Retrieval-Augmented Generation (RAG) pipeline using:
- **FAISS** for vector similarity search
- **Sentence-Transformers** for embedding text
- **Gemini 2.0 Flash** for AI-powered answer generation

Designed for simplicity and learning, this project works with a single large text file to answer questions grounded in your data.

---

## 📁 Project Structure

```
rag/
│
├── data.txt             # Your knowledge source (large text file)
├── embedder.py          # Converts text chunks into embeddings
├── vector_store.py      # Builds and queries the FAISS index
├── retriever.py         # Retrieves relevant text chunks using FAISS
├── generator.py         # Generates answers using Gemini API
├── prompt.py            # Formats the final prompt for Gemini
├── rag_pipeline.py      # Ties all components into a complete pipeline
├── app.py               # CLI interface to ask questions
├── .env                 # Stores your GOOGLE_API_KEY securely
└── requirements.txt     # Required Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TechManTejas/rag.git
cd rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key
```

> You can get your Gemini API key from: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

### 4. Add your data

Put your large text into `data.txt`. This will be chunked, embedded, and indexed.

### 5. Run the app

```bash
python app.py
```

Type your question and get AI-generated answers using your data.

---

## 🧠 How it works

1. **Embedding**: Your text is split into chunks and converted into vectors using SentenceTransformers.
2. **Storage**: The vectors are indexed using FAISS for fast similarity search.
3. **Retrieval**: A user query is embedded and top-k similar chunks are retrieved.
4. **Generation**: The query and retrieved context are sent to Gemini, which generates a natural answer.

---

## 🛠 Requirements

- Python 3.8+
- Internet access (to call Gemini API)

---

## 📃 License

MIT License