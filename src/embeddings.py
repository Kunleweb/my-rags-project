from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages the generation of document embeddings using the SentenceTransformer library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Initializes the SentenceTransformer model if not already loaded."""
        if self.model is not None:
            return

        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if self.model is None:
            raise ValueError("Embedding model not loaded. Call `load_model()` first.")

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

