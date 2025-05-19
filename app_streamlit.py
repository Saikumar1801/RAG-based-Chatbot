# app_streamlit.py

import streamlit as st
import os
from dotenv import load_dotenv # To ensure .env is loaded for rag_chatbot functions

# Import functions from your main RAG script
from rag_chatbot import (
    load_api_key as rag_load_api_key, # Rename to avoid conflict if st also has load_api_key
    load_documents,
    split_documents,
    create_vector_store,
    setup_rag_pipeline
)

# --- 1. Page Configuration and API Key Loading ---
st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 RAG Chatbot with LangChain & OpenAI")
st.caption("Query your knowledge base with AI-powered answers.")

# Load API Key - Crucial first step
# This needs to happen before any OpenAI calls are made by imported functions
try:
    # Ensure .env variables are loaded if not already set in the environment
    # This is important if rag_chatbot.py's load_api_key isn't called directly by Streamlit's execution flow
    # or if Streamlit runs in an environment where .env isn't automatically picked up for sub-modules.
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv() # Load .env if key not already in environment
    
    # Now, explicitly call the API key loading/validation logic from rag_chatbot
    rag_load_api_key() # This will print "OpenAI API Key loaded successfully." or raise ValueError
    st.sidebar.success("OpenAI API Key loaded successfully!")
except ValueError as e:
    st.error(f"🚨 Critical Error: {e}. The application cannot start without a valid API key.")
    st.warning("Please ensure your `OPENAI_API_KEY` is correctly set in a `.env` file in the project root, or as an environment variable if deploying.")
    st.markdown("""
    **How to fix:**
    1. Create a file named `.env` in the root directory of this project.
    2. Add your OpenAI API key to it like this: `OPENAI_API_KEY="your_actual_api_key_here"`
    3. Rerun the Streamlit app.
    If deploying (e.g., to Streamlit Community Cloud), set this as a secret environment variable.
    """)
    st.stop() # Halt execution if API key is missing

# --- 2. Global Variables & Constants ---
DATA_FILE_PATH = "knowledge_base.txt" # Default knowledge base file

# --- 3. Caching for Expensive Operations ---
# Cache the RAG pipeline setup to avoid re-computation on every interaction.
# `allow_output_mutation=True` can be necessary for complex objects like LangChain chains
# if their internal state might change in ways Streamlit's hasher doesn't detect.
# However, for read-only use of the chain after creation, it might not be strictly needed.
# `cache_resource` is generally preferred for models, connections, etc.
@st.cache_resource(show_spinner="Initializing RAG System...")
def initialize_rag_system(data_file_path):
    """
    Loads data, creates vector store, and sets up the RAG pipeline.
    This function is cached to run only once or when data_file_path changes.
    """
    st.write(f"Loading knowledge from: `{data_file_path}`") # For user feedback
    if not os.path.exists(data_file_path):
        st.error(f"Knowledge base file not found: {data_file_path}")
        return None

    try:
        documents = load_documents(data_file_path)
        if not documents:
            st.error("No documents were loaded. Check the knowledge base file.")
            return None
        
        text_chunks = split_documents(documents)
        if not text_chunks:
            st.error("Document splitting resulted in no chunks.")
            return None

        vector_store = create_vector_store(text_chunks)
        rag_chain = setup_rag_pipeline(vector_store)
        st.success("RAG system initialized successfully!")
        return rag_chain
    except Exception as e:
        st.error(f"Error during RAG system initialization: {e}")
        st.exception(e) # Shows full traceback in Streamlit for debugging
        return None

# --- 4. Initialize RAG Chain ---
# Check if knowledge base file exists before attempting to initialize
if not os.path.exists(DATA_FILE_PATH):
    st.error(f"Knowledge base file `{DATA_FILE_PATH}` not found. Please create it or check the path.")
    st.stop()

rag_chain = initialize_rag_system(DATA_FILE_PATH)

if not rag_chain:
    st.error("🔴 RAG pipeline initialization failed. The chatbot cannot function.")
    st.stop() # Stop if pipeline isn't ready

# --- 5. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today based on the loaded knowledge?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the knowledge base..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"query": prompt})
                answer = response.get("result", "Sorry, I encountered an issue and couldn't find an answer.")
                
                message_placeholder.markdown(answer)
                full_response_content = answer

                # Optionally display source documents
                if response.get("source_documents"):
                    with st.expander("🔍 View Sources"):
                        for i, doc in enumerate(response["source_documents"]):
                            source_name = doc.metadata.get('source', 'Unknown source')
                            st.markdown(f"**Source {i+1}:** `{source_name}`")
                            st.caption(doc.page_content)
                            st.divider()
            
            except Exception as e:
                st.error(f"An error occurred while fetching the answer: {e}")
                st.exception(e)
                full_response_content = "Sorry, I ran into a problem. Please try again."
                message_placeholder.markdown(full_response_content)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# --- 6. Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info(
    """
    This is a Retrieval-Augmented Generation (RAG) chatbot.
    It uses LangChain and OpenAI to answer questions based on
    the content of the `knowledge_base.txt` file.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Controls")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I help you?"}]
    st.rerun()

# Optional: Display knowledge base file name
st.sidebar.markdown("SK")
st.sidebar.subheader("Knowledge Base")
st.sidebar.markdown(f"Currently using: `{DATA_FILE_PATH}`")

