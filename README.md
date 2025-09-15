# Advanced Hybrid RAG

An advanced document question-answering (Q&A) system leveraging a hybrid Retrieval-Augmented Generation (RAG) architecture. This project enables querying complex PDFs containing tables, formulas, code snippets, and images with accurate, context-aware responses.

## Demo

Try the live demo deployed on Streamlit Cloud:  
[https://advanced-hybrid-rag.streamlit.app](https://advanced-hybrid-rag.streamlit.app)

## Features

- **Multi-format Document Ingestion:** Supports PDFs with rich content including text, tables, formulas, and images.
- **Hybrid Retrieval System:** Combines vector embeddings (Chroma with HuggingFace embeddings) and BM25 keyword retrieval for robust document search.
- **Contextual Question Reformulation:** Uses a language model to reformulate queries considering chat history for disambiguation.
- **History-Aware Conversational RAG:** Answers user questions based on retrieved document context and conversation history.
- **Clear Chat and Export:** Clear conversation history and export chat logs to JSON for later analysis.

## Technologies used

- Python 3.x
- [LangChain](https://github.com/langchain-ai/langchain) for retrieval chains and conversational AI workflows  
- [unstructured](https://github.com/Unstructured-IO/unstructured) for PDF parsing and document partitioning  
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for semantic text encoding  
- [Chroma](https://github.com/chroma-core/chroma) vector store for efficient similarity search  
- [Groq API](https://groq.com/) accessed via `langchain_groq` for large language model inference  
- Streamlit for the web user interface

## Getting Started

### Prerequisites

- Python 3.8+
- API keys for Groq and HuggingFace (can be user supplied via Streamlit UI)
- System libraries including but not limited to:
  - `libgl1-mesa-glx`
  - `poppler-utils`
  - `libglib2.0-0`
  - `libsm6`
  - `libxext6`
  - `libxrender1`
  - `tesseract-ocr`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/vipinkaulwar13/Advanced-Hybrid-RAG.git
cd Advanced-Hybrid-RAG
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure system dependencies are installed (see `packages.txt` for details if using Streamlit Cloud).

### Usage

1. Run the Streamlit app locally:

```bash
streamlit run Streamlit_App.py
```

2. Provide your GROQ and HuggingFace API keys in the sidebar.

3. Upload PDF documents and ask questions via the chat interface.

4. Export or clear chat history as needed.

## Architecture Overview

- **Ingestion:** PDFs are partitioned into elements (text, tables, images).
- **Processing:** Content is extracted, tables converted to text.
- **Chunking:** Text chunks are created for embedding.
- **Embedding & Indexing:** Embeddings via HuggingFace models; stored in Chroma vector store.
- **Hybrid Retrieval:** BM25 keyword retriever combined with vector retriever using weighted ensemble.
- **Question Answering:** A Groq-powered LLM answers queries using retrieved document context and chat history.
- **UI:** Streamlit frontend for file upload, key entry, question input, and conversation display.

---

Developed by Vipin Kaulwar  
[GitHub Repository](https://github.com/vipinkaulwar13/Advanced-Hybrid-RAG)  
[Live Demo](https://advanced-hybrid-rag.streamlit.app)
