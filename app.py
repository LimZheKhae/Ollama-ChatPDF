"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import ollama
import sqlite3

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List, Tuple, Dict, Any, Optional
from streamlit_pdf_viewer import pdf_viewer

# Global variable
vector_db = None
file_upload = None
# Define the database path
db_path = os.getcwd() + r'\db'

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
        
@st.cache_resource(show_spinner=True)
# Extract model names
def extract_model_names(_models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        _models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")

    # Validate structure
    if "models" not in _models_info or not isinstance(_models_info["models"], list):
        raise ValueError("Invalid `_models_info` structure: 'models' key missing or not a list.")

    # Safely extract model names from the 'model' attribute of each item
    model_names = tuple(model.model for model in _models_info["models"])

    logger.info(f"Extracted model names: {model_names}")
    return model_names

# Create vector database
def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()
    
    # Split the documents into chunks    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Updated embeddings configuration
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="myRAG",
        persist_directory=db_path,
    )
    
    logger.info("Vector DB created", temp_dir)
    # print(f"--------  DEBUG: Vector DB created = {vector_db}  ------------")
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

# Process user question
def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    print(f"--------  DEBUG: Processing question = {question}  ------------")
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

# Delete vector database
def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()

            logger.info("Deleted database files")
            
            # print(f"--------  *** DEBUG: Vector DB deleted  ------------")
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            # vector_db.close()
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except sqlite3.OperationalError as e:
            if "no such table: collections" in str(e):
                st.error("The collections table does not exist in the database.")
                logger.error(f"Error deleting collection: {e}")
            else:
                st.error(f"An error occurred while deleting the collection: {e}")
                logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)
    
    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "pdf_uploader_key" not in st.session_state:
        st.session_state["pdf_uploader_key"] = 0
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )

    # Regular file upload with unique key
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", 
        type="pdf", 
        accept_multiple_files=False,
        key=st.session_state["pdf_uploader_key"],
    )
    # Update the content in col1 with the new file name
    file_name = "📄 " + st.session_state["uploaded_files"][0].name if st.session_state["uploaded_files"] else ""
    col1.write(f" {file_name}")

    # Process uploaded PDF
    if file_upload:
        if st.session_state["vector_db"] is not None:
            # Delete the existing vector database
            delete_vector_db(st.session_state["vector_db"])
            st.session_state["pdf_pages"] = None
            st.session_state["file_uploaded"] = False
            # Clear all session state variables
            st.session_state.clear()    
        
        st.session_state["uploaded_files"] = [file_upload]
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)
                st.session_state["pdf_pages"] = file_upload
                st.session_state["pdf_uploader_key"] += 1  # Update the key to reset the uploader
                st.rerun()  # Rerun the app to clear the file upload state
            
    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                pdf_viewer(st.session_state["pdf_pages"].getvalue(), width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Reset", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        # Ensure the vector_db exists and call delete_collection
        if "vector_db" in st.session_state and st.session_state["vector_db"]:
            st.session_state["vector_db"].delete_collection()
        else:
            st.warning("Vector database is not initialized.")
        
        # Clear the session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["pdf_pages"] = None
        st.session_state["file_uploaded"] = False
        # Clear all session state variables
        st.session_state.clear()    
        st.success("File upload reset. Please upload a new file.")
        # Optionally, provide user feedback
        st.rerun()

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")


if __name__ == "__main__":
    main()