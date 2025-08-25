# run_once.py
# Description: This script processes PDF documents from a specified directory,
# splits them into text chunks, generates embeddings, and saves them into a
# FAISS vector store. This is a one-time setup script for the RAG chatbot.

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# Define the paths for the source documents and the vector store.
DOCS_PATH = "docs/"
VECTORSTORE_PATH = "vectorstore/"

def create_vectorstore():
    """
    Loads PDF documents, splits them into chunks, creates embeddings,
    and saves them to a FAISS vector store.
    """
    print("--- Starting Vector Store Creation ---")

    # 1. Load documents from the specified directory
    print(f"Loading documents from: {DOCS_PATH}")
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"Error: The directory '{DOCS_PATH}' is empty or does not exist.")
        print("Please add your PDF files to the 'docs/' directory and run again.")
        return

    loader = PyPDFDirectoryLoader(DOCS_PATH)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} documents.")

    # 2. Split the documents into smaller chunks for processing
    print("Splitting documents into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Initialize the embedding model
    # We use a sentence-transformer model for creating vector representations of the text.
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
    )
    print("Embedding model initialized.")

    # 4. Create the FAISS vector store from the document chunks and embeddings
    print("Creating FAISS vector store...")
    # This process can take some time depending on the number of documents.
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")

    # 5. Save the vector store locally
    print(f"Saving vector store to: {VECTORSTORE_PATH}")
    if not os.path.exists(VECTORSTORE_PATH):
        os.makedirs(VECTORSTORE_PATH)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("--- Vector Store Creation Complete ---")

if __name__ == "__main__":
    create_vectorstore()

