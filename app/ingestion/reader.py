import os
from docx import Document


def read_docx(file_path):
    document = Document(file_path)

    full_text = []

    for paragraph in document.paragraphs:
        full_text.append(paragraph.text)

    return "\n".join(full_text)


def read_folder(folder_path, category=None):
    documents = []
    
    if not os.path.isdir(folder_path):
        return documents

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".docx"):
            full_path = os.path.join(folder_path, file_name)
            text = read_docx(full_path)
            document = {
                "filename": file_name,
                "category": category or os.path.basename(folder_path),
                "text": text
            }
            documents.append(document)

    return documents


def read_project(project_path):
    all_documents = []

    if not os.path.exists(project_path):
        return all_documents

    for folder_name in os.listdir(project_path):
        folder_path = os.path.join(project_path, folder_name)

        if os.path.isdir(folder_path):
            documents = read_folder(folder_path, category=folder_name)
            all_documents.extend(documents)

    return all_documents
