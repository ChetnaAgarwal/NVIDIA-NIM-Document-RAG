# NVIDIA-NIM-Project
# Nvidia NIM Document QA System Project

This project is a document-based question-answering (QA) application using Streamlit and LangChain integrated with NVIDIA NIM for natural language understanding and document retrieval. By embedding documents into vector representations and utilizing NVIDIA’s LLM (Large Language Model) endpoints, users can query specific documents for context-driven responses.

# Features
Document Embedding: Supports loading documents from PDF files, chunking and embedding them for efficient retrieval.
NVIDIA NIM API Integration: Utilizes NVIDIA’s language models for high-quality, context-aware answers.
Streamlit UI: Provides a simple, interactive interface for asking questions and displaying relevant document content.
FAISS Vector Store: Uses FAISS for fast vector-based similarity search on embedded document chunks.


# STEPS TO RUN
# Step 1: Clone the Repository
Start by cloning the repository to your local machine:

git clone <repository-url>
cd <repository-directory>

# Step 2: Create a Virtual Environment
To keep dependencies isolated, create a virtual environment:

python3 -m venv venv
source venv/bin/activate    # On Windows, use venv\Scripts\activate

# Step 3: Create the .env File
The .env file will store the NVIDIA API key for accessing NVIDIA NIM. 

In the root directory of the project, create a new file named .env. Open the .env file and add your NVIDIA API key as shown below:

NVIDIA_API_KEY=<your_nvidia_api_key>

# Step 4: Install Dependencies

Install the required packages listed in requirements.txt:

pip install -r requirements.txt

# Step 5: Add Additional Files for RAG (Retrieval-Augmented Generation)

For Retrieval-Augmented Generation (RAG), ensure the following files and directories are set up:

- Document Directory:

Place your PDF files for QA in the directory ./us_census.
You can create this directory in the project root if it doesn’t already exist:

mkdir us_census
Add your PDF documents to this directory.

- Streamlit Application (app.py):

The primary application file app.py should be in the root directory.
This file initializes document embeddings, retrieves relevant documents, and generates answers.

# Step 6: Run the Application
To start the Streamlit application, use the following command:

streamlit run app.py
Upon starting, the application will open in your browser, and you can interact with the document QA system.
