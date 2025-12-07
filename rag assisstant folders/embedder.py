from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded!")

    def embed(self, texts):
        # texts: list of strings
        return self.model.encode(texts, show_progress_bar=True)

    def get_embedding(self, text):
        # text: single string
        return self.model.encode([text], show_progress_bar=False)[0]
