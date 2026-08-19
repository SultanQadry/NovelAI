from app.ingestion.reader import read_project
from app.processing.processor import DocumentProcessor
from app.processing.chunker import chunk_text
from app.embeddings.embedder import Embedder
from app.database.chroma_store import ChromaStore
import sys

# Configure stdout to handle Unicode characters (e.g. Arabic) on Windows
sys.stdout.reconfigure(encoding="utf-8")


def main():

    print("=" * 60)
    print("NovelAI - Testing Document Processing")
    print("=" * 60)

    documents = read_project("data/raw")

    print(f"\nTotal documents found: {len(documents)}")
    print("=" * 60)

    processor = DocumentProcessor()

    embedder = Embedder()

    chroma_store = ChromaStore()

    # Reset collection to remove old poorly-chunked data
    chroma_store.reset_collection()

    for i, document in enumerate(documents, start=1):

        processed_document = processor.process(document)

        print(f"\nDocument #{i}")
        print(f"Filename : {processed_document['filename']}")
        print(f"Category : {processed_document['category']}")
        print(f"Characters: {len(processed_document['text'])}")
        print(f"Words: {len(processed_document['text'].split())}")

        # Chunk the document
        chunks = chunk_text(processed_document["text"])

        print(f"Number of chunks: {len(chunks)}")

        docs_to_store = []
        embeddings_to_store = []

        for chunk_number, chunk in enumerate(chunks, start=1):

            chunk_id = f"{processed_document['filename']}_chunk_{chunk_number}"
            chunk_metadata = {
                "filename": processed_document["filename"],
                "category": processed_document["category"],
                "chunk_number": chunk_number
            }
            
            # Inject Document Context to ensure every chunk retains the semantic meaning of the document
            doc_title = processed_document['filename'].replace('.docx', '').replace('.txt', '')
            contextualized_chunk = f"Document Title: {doc_title}\nCategory: {processed_document['category']}\n\n{chunk}"

            # Embed each chunk
            embedding = embedder.embed_document(contextualized_chunk)

            docs_to_store.append({
                "id": chunk_id,
                "text": contextualized_chunk,
                "metadata": chunk_metadata
            })
            embeddings_to_store.append(embedding)

            print(f"\nChunk {chunk_number}")
            print("-" * 40)
            print(chunk[:300])
            print("-" * 40)

        # Store all embedded chunks for the current document in ChromaDB
        if docs_to_store:
            chroma_store.add_documents(docs_to_store, embeddings_to_store)

    print(f"\nTotal chunks stored in ChromaDB: {chroma_store.count()}")


if __name__ == "__main__":
    main()
