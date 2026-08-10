from app.ingestion.reader import read_project
from app.processing.processor import DocumentProcessor
import app.ingestion.reader as reader
import sys

# Configure stdout to handle Unicode characters (e.g. Arabic) on Windows
sys.stdout.reconfigure(encoding='utf-8')

print(reader.__file__)
print(dir(reader))


def main():

    print("=" * 60)
    print("NovelAI - Testing Document Ingestion")
    print("=" * 60)

    documents = read_project("data/raw")

    print(f"\nTotal documents found: {len(documents)}")
    print("=" * 60)

    processor = DocumentProcessor()

    for i, document in enumerate(documents, start=1):

        processed_document = processor.process(document)

        print(f"\nDocument #{i}")
        print(f"Filename : {processed_document['filename']}")
        print(f"Category : {processed_document['category']}")
        print(f"Characters: {len(processed_document['text'])}")
        print("-" * 60)
        print(processed_document["text"][:200])
        print("-" * 60)


if __name__ == "__main__":
    main()