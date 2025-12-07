from embedder import Embedder
from faiss_stores import FAISSStore

class Retriever:
    def __init__(self, vector_dim=384):
        self.embedder = Embedder()
        self.store = FAISSStore(vector_dim=vector_dim)

    def embed_query(self, text):
        return self.embedder.embed([text])[0]

    def get_relevant_chunks(self, query, k=5):
        # Create embedding for query
        query_vector = self.embedder.get_embedding(query)

        # Search FAISS
        return self.store.search(query_vector, k=k)
