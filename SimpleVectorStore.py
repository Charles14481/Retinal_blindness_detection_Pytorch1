"""
Vector store service for RAG system
Handles embeddings generation, storage, and similarity search using FAISS
"""

import json
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional

class SimpleVectorStore:
    def __init__(self, dimension: int = 1536):
        """
        Initialize vector store

        Args:
            dimension: Dimension of embedding vectors (OpenAI embeddings are 1536)
        """
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        self.index_to_id = {}
        self.next_id = 0

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar vectors"""
        if not self.vectors:
            return []

    # Calculate cosine similarity
        similarities = []
        query_norm = np.linalg.norm(query_vector)

        for i, vector in enumerate(self.vectors):
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0 or query_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
            similarities.append((i, similarity))

    # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
        results = []
        for i, (idx, score) in enumerate(similarities[:k]):
            results.append((self.metadata[idx], score))

        return results

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts using OpenAI API

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not available")

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        all_embeddings = []
        batch_size = 100  # OpenAI API limit

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            data = {
                "input": batch_texts,
                "model": "text-embedding-ada-002"
            }

            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()

                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(embeddings)

                print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                all_embeddings.extend([[0.0] * 1536 for _ in batch_texts])

        return all_embeddings