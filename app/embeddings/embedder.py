from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Converts text into numerical vectors using multilingual E5.
    """

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small"
        )

        print("Embedding model ready")

    def embed_document(self, text):
        """
        Create an embedding for a document chunk.
        """

        text = f"passage: {text}"

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_query(self, text):
        """
        Create an embedding for a user search query.
        """

        text = f"query: {text}"

        return self.model.encode(
            text,
            normalize_embeddings=True
        )