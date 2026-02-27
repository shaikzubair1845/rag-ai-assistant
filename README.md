# 📚 RAG AI Assistant (FAISS-Based)

A Retrieval-Augmented Generation (RAG) based AI Assistant that reads PDF documents, generates embeddings, stores them in a FAISS vector database, and answers user queries using OpenAI GPT models.

This project implements a complete end-to-end RAG pipeline from scratch using Python.

---

## 🚀 Project Overview

This AI assistant allows users to:

- 📄 Load and process PDF documents
- 🧹 Clean and preprocess extracted text
- ✂️ Split text into overlapping chunks
- 🔎 Generate embeddings using Sentence Transformers
- 🗂 Store embeddings in FAISS vector database
- 🤖 Retrieve relevant context using semantic search
- 💬 Generate accurate answers using OpenAI GPT-4o-mini
- 🔁 Interactively ask questions via terminal

---

## 🧠 Architecture

PDF → Text Extraction → Cleaning → Chunking  
↓  
Embedding Generation (Sentence Transformers)  
↓  
FAISS Vector Storage  
↓  
Query Embedding  
↓  
Top-K Retrieval  
↓  
LLM (GPT-4o-mini)  
↓  
Final Answer  

---

## 🛠 Tech Stack

- Python
- PyMuPDF (fitz) – PDF text extraction
- Regex / re – Text cleaning
- Custom Chunking Logic – Overlapping text splitting
- Sentence Transformers (all-MiniLM-L6-v2) – Embedding generation
- FAISS (IndexFlatL2) – Vector similarity search
- OpenAI GPT-4o-mini – Answer generation
- NumPy – Vector processing

---

## 📂 Project Structure

rag-ai-assistant/
│
├── main.py              # Entry point
├── pdf_loaders.py       # PDF extraction
├── text_cleaners.py     # Text preprocessing
├── chunker.py           # Text chunking logic
├── embedder.py          # Embedding generation
├── faiss_stores.py      # FAISS vector storage
├── retriever.py         # Semantic retrieval
├── rag_pipeline.py      # RAG + OpenAI integration
├── requirements.txt

---

## ⚙️ How It Works

1. The PDF is loaded and converted into raw text.
2. Text is cleaned and normalized.
3. Text is split into overlapping chunks.
4. Each chunk is converted into embeddings.
5. Embeddings are stored in FAISS.
6. When a question is asked:
   - The query is embedded.
   - FAISS retrieves top-k similar chunks.
   - Retrieved context is passed to GPT model.
   - The final answer is generated.

---

## ▶️ Installation & Setup

### 1️⃣ Clone the repository

git clone <your-repo-link>  
cd rag-ai-assistant  

### 2️⃣ Install dependencies

pip install -r requirements.txt  

### 3️⃣ Add your OpenAI API key

Inside `main.py`, replace:

OPENAI_API_KEY = "your_api_key_here"

---

## ▶️ Run the Project

python main.py  

If vector database doesn't exist, it will automatically build one.

Then start asking questions about the PDF.

Type `exit` to stop.

---

## 🔥 Key Highlights

- End-to-end RAG pipeline built from scratch
- Custom FAISS integration (no external vector DB service)
- Modular architecture
- Interactive CLI interface
- Optimized for semantic search accuracy

---

## 📌 Future Improvements

- Add Streamlit / Flask UI
- Deploy on cloud
- Add support for multiple PDFs
- Add re-ranking mechanism
- Add caching for faster responses

---

## 👨‍💻 Author

Shaik Zubair  
Aspiring Data Scientist  
Hyderabad, India
