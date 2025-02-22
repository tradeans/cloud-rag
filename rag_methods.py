import os
import logging
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Please add it to the .env file.")

# Function to get the appropriate loader for the document type
def get_loader(file_path):
    """ Returns the appropriate document loader based on file type """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".json"):
        loader = load_json(file_path)
    else:
        raise ValueError(f"Document type {file_path} is not supported.")
    
    # Ensure loader returns documents in iterable format
    documents = loader.load() if hasattr(loader, 'load') else []
    print(f"Loaded {len(documents)} documents from {file_path}")  # Log the number of documents loaded
    return documents

def load_json(file_path):
    """ Load and extract relevant content from a JSON file. """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Read JSON file
            # Assuming the JSON contains a key "text" where relevant information is stored
            if isinstance(data, dict):  # Single JSON object
                text_content = data.get("text", "")
            elif isinstance(data, list):  # JSON array
                text_content = "\n".join([item.get("text", "") for item in data if isinstance(item, dict)])
            else:
                text_content = ""
            print(f"Loaded content from JSON file {file_path}: {text_content[:100]}...")  # Print a snippet of the content
            return [{"page_content": text_content, "source": file_path}]
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def process_documents(file_paths):
    """ Processes documents (PDF, DOCX, TXT, JSON) into vector embeddings. """
    docs = []
    for file_path in file_paths:
        docs.extend(get_loader(file_path))  # Load documents based on file type

    if len(docs) == 0:
        raise ValueError("No valid documents loaded. Please check the files in the docs folder.")
    
    print(f"Loaded {len(docs)} documents in total.")  # Log total number of documents loaded

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings(openai_api_key= api_key)  # Provide OpenAI embeddings

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=3000)
    document_chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(document_chunks)} chunks.")  # Log number of document chunks

    if len(document_chunks) == 0:
        raise ValueError("No valid document chunks created. Please check your documents.")

    # Create the vector database (Chroma)
    vector_db = Chroma.from_documents(document_chunks, embedding=embeddings)
    return vector_db

