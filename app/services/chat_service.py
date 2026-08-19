class ChatService:

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def chat(self, query, history=None):

        chunks = self.retriever.search(
            query,
            top_k=7
        )

        answer = self.generator.generate(
            query=query,
            chunks=chunks,
            history=history
        )

        return {
            "answer": answer,
            "chunks": chunks
        }