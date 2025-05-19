# RAG Chatbot Assignment: Innovatech Solutions Assistant

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Python, LangChain, and OpenAI. The chatbot can answer questions based on a provided knowledge base, specifically information about "Innovatech Solutions" contained in `knowledge_base.txt`. It includes both a command-line interface (CLI) and a Streamlit web application for interaction.

## Assignment Objective

This assignment aims to assess skills in Retrieval-Augmented Generation (RAG), Python, and LangChain for building a Chat application. The task is to create a RAG-based Chatbot that uses external data sources to answer questions accurately and contextually.

## Project Overview

The core of this project is a RAG pipeline built with LangChain:

1.  **Data Loading (Task 1):** A text file (`knowledge_base.txt`) serves as the external data source. This file is loaded and processed.
2.  **Set Up RAG with LangChain (Task 2):**
    *   The loaded documents are split into manageable chunks.
    *   These chunks are then embedded using OpenAI's embedding models.
    *   A FAISS vector store is created to efficiently search these embeddings.
    *   A LangChain `RetrievalQA` chain is set up, combining a retriever (from the vector store) and a language model (OpenAI's GPT) to answer questions based on retrieved context.
3.  **Build the Chatbot (Task 3):**
    *   A command-line interface (`rag_chatbot.py`) allows users to interact with the RAG pipeline.
    *   (Bonus) A Streamlit web application (`app_streamlit.py`) provides a user-friendly graphical interface for the chatbot.

## Features

*   **Dynamic Knowledge Base:** Easily update or change the `knowledge_base.txt` file to alter the chatbot's knowledge.
*   **Contextual Answers:** The RAG pipeline ensures answers are grounded in the provided documents, reducing hallucinations.
*   **Source Referencing:** The chatbot can indicate which parts of the knowledge base were used to generate an answer (visible in CLI and Streamlit).
*   **Interactive CLI:** A simple command-line interface for direct interaction.
*   **Streamlit Web App (Bonus):** A user-friendly web interface for a better chat experience.
*   **Modular Code:** Functions are organized for clarity and reusability in `rag_chatbot.py`.

## Technology Stack

*   **Python 3.8+**
*   **LangChain:** For building the RAG pipeline (document loading, splitting, vector stores, chains, LLM integration).
*   **OpenAI API:** For text embeddings and language model capabilities (GPT-3.5-turbo).
*   **FAISS:** For efficient similarity search in the vector store.
*   **Streamlit:** For building the interactive web application (bonus).
*   **python-dotenv:** For managing environment variables (API keys).

## Prerequisites

*   Python 3.8 or higher.
*   `pip` (Python package installer).
*   An OpenAI API Key.
*   Git (for cloning the repository).

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Saikumar1801/RAG-based-Chatbot.git
    cd RAG-based-Chatbot
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file should be present in the repository.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, you can create one or install packages manually:
    ```bash
    pip install langchain langchain-openai langchain-community faiss-cpu openai python-dotenv streamlit tiktoken
    ```
    *(Note: `faiss-cpu` is for CPU-only. If you have a compatible GPU and CUDA setup, you might consider `faiss-gpu`.)*

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

5.  **Prepare the Knowledge Base:**
    Ensure the `knowledge_base.txt` file is present in the root directory. You can modify its content to change the chatbot's knowledge. The current file contains information about "Innovatech Solutions."

## Running the Application

### 1. Command-Line Interface (CLI)

To run the chatbot via the command line:
```bash
python rag_chatbot.py
```
You will be prompted to enter your questions. Type `exit` or `quit` to end the session.

### 2. Streamlit Web Application (Bonus)

To run the Streamlit web application:
```bash
streamlit run app_streamlit.py
```
This will typically open the application in your default web browser (e.g., at `http://localhost:8501`).

## Project Structure

```
.
├── .env                  # Stores environment variables (e.g., API keys) - GITIGNORED
├── knowledge_base.txt    # The data source for the RAG chatbot
├── rag_chatbot.py        # Main Python script for RAG pipeline and CLI
├── app_streamlit.py      # Python script for the Streamlit web application (Bonus)
├── sample_qa_responses.txt # Sample questions and chatbot responses
├── requirements.txt      # Python package dependencies
└── README.md             # This file
```
