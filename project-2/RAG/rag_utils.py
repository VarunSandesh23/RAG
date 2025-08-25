# rag_utils.py
# Description: This module contains the core functions for the RAG (Retrieval-Augmented Generation)
# chatbot. It handles loading the vector store, initializing the language model,
# and setting up the conversational retrieval chain.

import os
from dotenv import load_dotenv # <-- ADD THIS LINE
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore/"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Groq API Configuration ---
load_dotenv() # <-- ADD THIS LINE TO LOAD .env FILE

# IMPORTANT: Make sure to set your Groq API key in a .env file
# or as an environment variable named 'GROQ_API_KEY'.
try:
    groq_api_key = os.environ['GROQ_API_KEY']
except KeyError:
    print("ERROR: GROQ_API_KEY environment variable not set.")
    print("Please create a .env file with GROQ_API_KEY='your_key' or set the environment variable.")
    exit()


def load_vectorstore():
    """
    Loads the FAISS vector store from the local path.

    Returns:
        FAISS: The loaded vector store object, or None if it fails.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"Error: Vector store not found at '{VECTORSTORE_PATH}'.")
        print("Please run 'run_once.py' first to create the vector store.")
        return None

    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
    return vectorstore

def create_rag_chain(vectorstore):
    """
    Creates the Retrieval-Augmented Generation (RAG) chain.

    Args:
        vectorstore (FAISS): The vector store containing document embeddings.

    Returns:
        RetrievalQA: The configured RAG chain.
    """
    if vectorstore is None:
        return None

    print("Creating RAG chain...")
    # Initialize the language model
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='llama3-8b-8192'
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3} # Retrieve top 3 most similar chunks
    )

    # Define the prompt template
    prompt_template = """
    You are a helpful assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide a concise and informative answer.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    rag_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    print("RAG chain created successfully.")
    return rag_chain

def get_answer(rag_chain, query):
    """
    Gets an answer from the RAG chain for a given query.

    Args:
        rag_chain (RetrievalQA): The RAG chain.
        query (str): The user's question.

    Returns:
        dict: A dictionary containing the answer and source documents.
    """
    if rag_chain is None:
        return {"query": query, "result": "Error: RAG chain is not initialized."}
        
    print(f"Invoking RAG chain for query: '{query}'")
    response = rag_chain.invoke({"query": query})
    print("RAG chain invocation complete.")
    return response
