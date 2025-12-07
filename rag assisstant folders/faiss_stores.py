import faiss
import numpy as np
import os
import pickle

class FAISSStore:
    def __init__(self, vector_dim, store_path="vectors/index.faiss", meta_path="vectors/metadata.pkl"):
        self.vector_dim = vector_dim
        self.store_path = store_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(vector_dim)
        self.metadata = []

        if os.path.exists(store_path) and os.path.exists(meta_path):
            self.load()

    def add(self, vectors, chunks):
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        self.metadata.extend(chunks)
        self.save()

    def save(self):
        faiss.write_index(self.index, self.store_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.store_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_vector, k=5):
        query_vector = np.array(query_vector).astype('float32')
        D, I = self.index.search(np.array([query_vector]), k)

        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((score, self.metadata[idx]))

        return results
