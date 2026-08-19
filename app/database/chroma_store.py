import chromadb

class ChromaStore:
    """
    Handles storing and retrieving document chunks
    using ChromaDB.
    """

    def __init__(self, persist_directory="data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="novel_documents"
        )

        print("ChromaDB ready")

    def add_documents(self, documents, embeddings):
        ids = []
        texts = []
        metadatas = []

        for i, document in enumerate(documents):

            ids.append(document["id"])
            texts.append(document["text"])
            metadatas.append(document["metadata"])

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored {len(documents)} documents in ChromaDB")

    def count(self):
        return self.collection.count()

    def search(self, query_embedding, n_results=3):
        """
        Search ChromaDB for the most relevant document chunks.
        """

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    def reset_collection(self):
        """
        Delete the current collection and create a fresh one.
        """

        self.client.delete_collection(name="novel_documents")

        self.collection = self.client.get_or_create_collection(
            name="novel_documents"
        )

        print("ChromaDB collection reset")