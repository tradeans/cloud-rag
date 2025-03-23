# fastapi_backend

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Initialize FastAPI
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (You can replace with specific origin if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

load_dotenv()

# Environment variables and OpenAI setup
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Please add it to the .env file.")

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key= api_key)

# Pydantic model for query input
class Query(BaseModel):
    query: str

# Function to get the appropriate loader for the document type
def get_loader(file_path):
    """ Returns the appropriate document loader based on file type """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Document type {file_path} is not supported.")
    return loader.load()

# Function to process documents into vector embeddings
def process_documents(file_paths):
    docs = []
    for file_path in file_paths:
        docs.extend(get_loader(file_path))  # Load documents based on file type

    if len(docs) == 0:
        raise ValueError("No valid documents loaded. Please check the files in the docs folder.")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=3000)
    document_chunks = text_splitter.split_documents(docs)
    vector_db = Chroma.from_documents(document_chunks, embedding=embeddings)
    return vector_db

# Function to get answer based on query
def get_answer_from_query(vector_db, query):
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    relevant_documents = retriever.get_relevant_documents(query)
    response = llm.predict(f"""
    You are an assistant for ISAT University. The user will ask questions related to the university. 
    Your task is to answer their exact question using the information provided by the following documents.

    The documents provided are:
    {relevant_documents}

    User's Question: {query}

    Based on the documents above, provide a clear and concise answer to the user's question.
    """)
    return response

# Endpoint for querying the model
@app.post("/query/")
async def query_endpoint(query: Query):
    try:
        # Define the path to the docs folder (assuming documents are pre-existing in the "docs" folder)
        doc_folder = "docs"  # Adjust if your documents are in a different directory
        # Get all document file paths in the docs folder
        file_paths = [os.path.join(doc_folder, file) for file in os.listdir(doc_folder) if os.path.isfile(os.path.join(doc_folder, file))]
        
        # Process documents into a vector database
        vector_db = process_documents(file_paths)
        
        # Retrieve the answer based on the user's query
        answer = get_answer_from_query(vector_db, query.query)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing query: {e}")
    
