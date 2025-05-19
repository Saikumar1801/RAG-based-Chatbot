# rag_chatbot.py

import os
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader # For CSV
# from langchain_community.document_loaders import PyPDFLoader # For PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback # For detailed error reporting

def load_api_key():
    """
    Loads the OpenAI API key from a .env file.
    Raises:
        ValueError: If the OPENAI_API_KEY is not found.
    """
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found. Please ensure it's set in your .env file or environment variables.")
    # The OpenAI library typically picks up the key automatically if it's set as an environment variable.
    # No explicit os.environ["OPENAI_API_KEY"] = openai_api_key is usually needed here.
    print("OpenAI API Key loaded successfully.")

def load_documents(file_path="knowledge_base.txt"):
    """
    Loads documents from the specified file path.
    Currently supports .txt files. Can be extended for .csv, .pdf, etc.

    Args:
        file_path (str): The path to the data file.

    Returns:
        list: A list of loaded documents.

    Raises:
        ValueError: If the file type is unsupported.
        FileNotFoundError: If the file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    # Determine file type and choose appropriate loader
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    # Example for CSV:
    # elif file_path.endswith(".csv"):
    #     loader = CSVLoader(file_path, encoding="utf-8")
    # Example for PDF (requires pip install pypdf):
    # elif file_path.endswith(".pdf"):
    #     loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Please use .txt, .csv, or .pdf")

    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {file_path}.")
    # print(f"Preview of first document content: {documents[0].page_content[:200]}...") # Optional
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits the loaded documents into smaller chunks for processing.

    Args:
        documents (list): A list of documents to split.
        chunk_size (int): The maximum number of characters per chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks (documents).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # Simpler splitting
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split input into {len(texts)} text chunks.")
    # print(f"Preview of first chunk: {texts[0].page_content[:100]}...") # Optional
    return texts

def create_vector_store(text_chunks, embeddings_model_name="text-embedding-ada-002"):
    """
    Creates a FAISS vector store from text chunks using OpenAI embeddings.

    Args:
        text_chunks (list): A list of text chunks (documents).
        embeddings_model_name (str): The name of the OpenAI embeddings model to use.

    Returns:
        FAISS: The created FAISS vector store.
    """
    print(f"Initializing embeddings model: {embeddings_model_name}...")
    embeddings = OpenAIEmbeddings(model=embeddings_model_name) # Requires OPENAI_API_KEY

    print(f"Creating vector store with {len(text_chunks)} chunks. This might take a moment...")
    # This step involves calling the OpenAI API to get embeddings for each chunk.
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    print("Vector store created successfully.")
    return vectorstore

def setup_rag_pipeline(vectorstore, llm_model_name="gpt-3.5-turbo", temperature=0.7, k_retriever=3):
    """
    Sets up the RAG (Retrieval-Augmented Generation) pipeline.

    Args:
        vectorstore (FAISS): The vector store containing document embeddings.
        llm_model_name (str): The OpenAI LLM model to use for generation.
        temperature (float): Controls the randomness of the LLM's output.
        k_retriever (int): The number of relevant documents to retrieve.

    Returns:
        RetrievalQA: The configured RAG QA chain.
    """
    print(f"Initializing LLM: {llm_model_name}...")
    llm = ChatOpenAI(
        model_name=llm_model_name,
        temperature=temperature
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retriever})
    print(f"Retriever created (k={k_retriever}).")

    # Define a prompt template to guide the LLM
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and helpful. If the context is empty or not relevant, state that you cannot answer based on the provided information.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    # chain_type="stuff": Puts all retrieved chunks directly into the prompt.
    # Other types like "map_reduce", "refine", "map_rerank" exist for different strategies.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True  # Crucial for seeing which documents were used
    )
    print("RAG QA chain created successfully.")
    return qa_chain

def run_chatbot(rag_chain):
    """
    Runs an interactive command-line chatbot session.

    Args:
        rag_chain (RetrievalQA): The RAG chain to use for answering questions.
    """
    print("\n--- Innovatech Chatbot Initialized ---")
    print("Ask me questions about the content in the knowledge base. Type 'exit' or 'quit' to end.")
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        if not user_query:
            continue

        try:
            # Get the response from the RAG chain
            # The .invoke method is part of the LangChain Expression Language (LCEL)
            response = rag_chain.invoke({"query": user_query})

            print(f"\nChatbot: {response['result']}")

            # Optionally print source documents for debugging/transparency
            if response.get("source_documents"):
                print("\n  --- Source Documents Used ---")
                for i, doc in enumerate(response["source_documents"]):
                    source_name = doc.metadata.get('source', 'Unknown source')
                    content_preview = doc.page_content[:150].replace('\n', ' ') + "..."
                    print(f"    Source {i+1} ({source_name}):\n      \"{content_preview}\"")
                print("  ---------------------------")

        except Exception as e:
            print(f"Chatbot: Sorry, an error occurred while processing your request: {e}")
            print(traceback.format_exc()) # Print full traceback for debugging

# --- Main execution flow for command-line chatbot ---
if __name__ == "__main__":
    try:
        # 1. Load API Key
        load_api_key()

        # 2. Load Documents from the knowledge base file
        # Ensure 'knowledge_base.txt' is in the same directory or provide the full path.
        DATA_FILE_PATH = "knowledge_base.txt"
        documents = load_documents(DATA_FILE_PATH)

        if documents:
            # 3. Split Documents into Chunks
            text_chunks = split_documents(documents)

            # 4. Create Vector Store (Embeddings)
            # This will make API calls to OpenAI for embeddings, might take time for large docs.
            vector_store = create_vector_store(text_chunks)

            # 5. Setup RAG Pipeline
            rag_qa_chain = setup_rag_pipeline(vector_store)

            # 6. Run the interactive chatbot
            run_chatbot(rag_qa_chain)

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}. Please make sure the data file exists.")
    except ValueError as val_error: # Handles API key errors and unsupported file types
        print(f"Configuration Error: {val_error}")
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())