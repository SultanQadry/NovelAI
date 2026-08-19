import chromadb
import sys
sys.stdout.reconfigure(encoding="utf-8")

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="novel_documents"
)

print("=" * 60)
print("ChromaDB Verification")
print("=" * 60)

print(f"Total records: {collection.count()}")

data = collection.get(
    limit=25,
    include=["documents", "metadatas", "embeddings"]
)

print(f"\nIDs found: {len(data['ids'])}")

for i in range(len(data["ids"])):

    print("\n" + "-" * 60)

    print(f"ID: {data['ids'][i]}")

    print(f"Text exists: {data['documents'][i] is not None}")
    print(f"Metadata exists: {data['metadatas'][i] is not None}")
    print(f"Embedding exists: {data['embeddings'][i] is not None}")

    if data["metadatas"][i]:
        print(f"Metadata: {data['metadatas'][i]}")

    if data["embeddings"][i] is not None:
        print(f"Embedding dimensions: {len(data['embeddings'][i])}")

print("\n" + "=" * 60)
