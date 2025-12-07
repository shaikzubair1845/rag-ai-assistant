import os
import pdf_loaders
import text_cleaners
import chunker
import embedder
import faiss_stores
import rag_pipeline  

PDF_PATH = r"your pdf file.pdf" #add your pdf file
VECTOR_DIM = 384
OPENAI_API_KEY = "abc api key"  # replace with your OpenAI key
MODEL_NAME = "gpt-4o-mini"  

def build_vector_database(pdf_path):
    print("\nLoading PDF...")
    loader = pdf_loaders.PDFLoader()
    raw_text = loader.load(pdf_path)
    print("PDF Loaded!")

    print("\nCleaning text...")
    cleaner = text_cleaners.TextCleaner()
    clean_text = cleaner.clean(raw_text)
    print("Text cleaned!")

    print("\nChunking text...")
    chunker_obj = chunker.TextChunker(chunk_size=500, overlap=50)
    chunks = chunker_obj.split(clean_text)
    print(f"Created {len(chunks)} chunks!")

    print("\nEmbedding chunks...")
    embedder_obj = embedder.Embedder()
    vectors = [embedder_obj.get_embedding(chunk) for chunk in chunks]
    print("Embeddings created!")

    print("\nSaving FAISS database...")
    store = faiss_stores.FAISSStore(vector_dim=VECTOR_DIM)
    store.add(vectors, chunks)
    print("Vector store saved!")

    print("\nDatabase build completed!")


def ask_question():
    # Initialize RAG with OpenAI GPT‑4o‑mini
    rag = rag_pipeline.RAGPipeline(
        api_key=OPENAI_API_KEY,
        model_name=MODEL_NAME,
        vector_dim=VECTOR_DIM
    )

    print("\nAsk any question about the PDF (type 'exit' to stop):\n")
    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag.generate_answer(query)
        print("\nAI:", answer, "\n")


if __name__ == "__main__":
    # Check if vector database exists
    if not (os.path.exists("vectors/index.faiss") and os.path.exists("vectors/metadata.pkl")):
        print("No vector database found. Building one...")
        build_vector_database(PDF_PATH)
    else:
        print("Vector database already exists. Skipping building step.")

    # Start interactive Q&A
    ask_question()
