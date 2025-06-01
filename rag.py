# Simplified RAG Pipeline Example
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MedicalRAGSystem:
    def __init__(self):
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.embeddings = None
        self.index = None

    def add_medical_documents(self, documents):
        """Add medical documents to the knowledge base"""
        self.knowledge_base.extend(documents)

        # Create embeddings for all documents
        embeddings = self.embedding_model.encode(documents)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        # Build FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)

        # Return relevant documents with scores
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base):
                relevant_docs.append({
                    'content': self.knowledge_base[idx],
                    'score': float(score),
                    'rank': i + 1
                })

        return relevant_docs
