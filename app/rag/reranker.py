from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks documents using a Cross-Encoder
    to improve semantic relevance between Query and Context.
    """

    def __init__(self, model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        print(f"Loading Reranker model ({model_name})...")
        self.model = CrossEncoder(model_name)
        print("Reranker model ready")

    def rerank(self, query, candidates, top_k=3, score_threshold=-2.0):
        """
        Scores candidate chunks by passing (query, chunk) pairs into the Cross-Encoder.
        Filters out low-scoring chunks and returns the best ones sorted by score.
        """
        if not candidates:
            return []

        # Prepare pairs for the Cross-Encoder
        pairs = [[query, candidate["text"]] for candidate in candidates]

        # Predict logits/scores (higher is better)
        scores = self.model.predict(pairs)

        # Attach scores to the candidate dictionaries
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # Filter out anything below the threshold
        # Note: Cross-Encoder logits can be negative, -2.0 is a safe permissive threshold for un-normalized logits
        valid_candidates = [
            c for c in candidates if c["rerank_score"] >= score_threshold]

        # Sort by best score descending
        valid_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return valid_candidates[:top_k]
