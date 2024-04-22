from rag.utils import benchmark

from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

# Returns a hugging face embedding model
def get_embedding_model(model_endpoint: str):
    return HuggingFaceHubEmbeddings(model=model_endpoint)

# Load PDF documents from file path
@benchmark
def load_documents(file_path: str):
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    return doc

# Split documents into chunks
@benchmark
def chunk_documents(doc, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
        )
    chunks = text_splitter.split_documents(doc)
    return chunks

# Save chunks into specified chromadb folder and returns the db
@benchmark
def save_chunks_to_chroma(chunks: list, embedding_model, chromadb_save_path: str):
    if os.path.exists(chromadb_save_path):
        print("ChromaDB already exists at the specified path. No changes made.")
    else:
        db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=chromadb_save_path)