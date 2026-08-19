from app.embeddings.embedder import Embedder
from app.database.chroma_store import ChromaStore
from app.rag.reranker import Reranker


class Retriever:
    """
    Retrieves the most relevant document chunks
    from ChromaDB for a given query, and reranks them.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.chroma_store = ChromaStore()
        self.reranker = Reranker()

        print("Retriever ready")

    def search(self, query, top_k=7, rerank_threshold=-3.5):
        """
        Search ChromaDB using semantic similarity, then use a Cross-Encoder
        to rerank the candidates for absolute precision.
        """

        # Retrieve a large pool of candidates for the reranker to evaluate
        fetch_k = max(top_k * 5, 25)

        # Convert query into an embedding
        query_embedding = self.embedder.embed_query(query)

        # Broad Search in ChromaDB
        results = self.chroma_store.search(
            query_embedding=query_embedding,
            n_results=fetch_k
        )

        candidates = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(documents)):
            candidates.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            })

        print(f"\n--- Candidate Pool Size: {len(candidates)} ---")

        # Rerank candidates using Cross-Encoder
        best_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            score_threshold=rerank_threshold
        )

        return best_results
        