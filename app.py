import streamlit as st
import os
from rag_methods import process_documents
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (GPT-4 model)
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key= api_key)

# Function to get all file paths in the 'docs' folder
def get_all_document_files(doc_folder="docs"):
    """Returns a list of all files in the specified folder."""
    return [os.path.join(doc_folder, f) for f in os.listdir(doc_folder) if os.path.isfile(os.path.join(doc_folder, f))]

# Correct function to retrieve relevant answers from the vector database
def get_answer_from_query(vector_db, query):
    # Retrieve the most relevant documents based on the query
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})  # k=4 retrieves top 4 relevant results
    relevant_documents = retriever.get_relevant_documents(query)

    # Use the language model to generate an answer based on the relevant documents
    response = llm.predict(f"""
    You are an assistant for ISAT University. The user will ask questions related to the university. 
    Your task is to answer their exact question using the information provided by the following documents.

    The documents provided are:
    {relevant_documents}

    User's Question: {query}

    Based on the documents above, provide a clear and concise answer to the user's question.
    """)

    return response

# Set up Streamlit interface
st.title("University Query Chatbot")

# Display the instructions
st.markdown("""
    ## Ask questions related to the university!
    Type in a question, and I'll provide an answer based on the documents I have.
""")

# Initialize the session state for the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to handle the chat interaction
def chat():
    user_input = st.text_input("You: ", key="input")
    if user_input:
        # Add the user input to the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get all documents in the "docs" folder
        doc_files = get_all_document_files("docs")  # Get all files in the docs folder

        # Process all the documents and retrieve an answer
        try:
            vector_db = process_documents(doc_files)  # Process all documents in the docs folder
            answer = get_answer_from_query(vector_db, user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    # Display the chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

# Display the chat box
chat()


    
