class DocumentProcessor:
    """
    Responsible for preparing documents
    before they enter the AI pipeline.
    """

    def __init__(self):
        print("Document Processor Ready")

    def _clean_text(self, text):
        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:
            line = line.strip()

            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def process(self, document):
        print(f"Processing document: {document['filename']}")

        cleaned_text = self._clean_text(document["text"])

        processed_document = {
            "filename": document["filename"],
            "category": document["category"],
            "text": cleaned_text
        }

        return processed_document
