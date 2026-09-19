import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kb")

def load_documents(kb_directory: str = KB_DIR) -> List[Document]:
    """Loads all markdown files from the specified knowledge base directory."""
    documents = []
    filepaths = glob.glob(os.path.join(kb_directory, "*.md")) + glob.glob(os.path.join(kb_directory, "*.txt"))
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        category = filename.replace(".md", "").replace(".txt", "")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "category": category,
                "path": filepath
            }
        )
        documents.append(doc)
    return documents

def get_chunked_documents(chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Loads knowledge base documents and splits them into smaller semantic chunks."""
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

if __name__ == "__main__":
    chunks = get_chunked_documents()
    print(f"Loaded {len(chunks)} chunks from knowledge base.")
    for i, c in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ({c.metadata['source']}) ---")
        print(c.page_content[:150] + "...")
