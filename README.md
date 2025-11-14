# AmbedkarGPT: RAG-based Q&A System A local, production-ready Retrieval-Augmented Generation (RAG) question-answering system using LangChain, ChromaDB, and Ollama. It answers questions based only on the content of Dr. B.R. Ambedkar's speech.

### Features Loads and chunks a document (speech.txt) for embedding

- Creates vector embeddings with HuggingFace and indexes with ChromaDB

- Uses local LLMs via Ollama (mistral:7b or similar) for answer generation

- RetrievalQA workflow: fetches most relevant chunks, generates context-anchored answers

### Setup Instructions

1.  Install Ollama Ollama is required for running local LLMs.
    

Linux/Mac:

bash curl -fsSL https://ollama.ai/install.sh | sh Windows: Download from ollama.ai/download

Pull your desired model (example: mistral:7b):

bash ollama pull mistral:7b Run Ollama server (keep this terminal open):

bash ollama serve

2.  Prepare Python Environment Create a new virtual environment:
    

bash python3 -m venv venv source venv/bin/activate # Linux/Mac venv\\Scripts\\activate # Windows Upgrade pip (optional, but recommended):

bash pip install --upgrade pip Install Python dependencies:

bash pip install -r requirements.txt NOTE: If you see warnings about "Ignoring invalid distribution …", follow the cleaning advice in this repository or recreate your venv.

3.  Add a Document Place speech.txt (the document you want to index) in the same folder as main.py.
    
4.  Run the Application bash python main.py On first run, embeddings and ChromaDB index are created (may take up to a minute).
    

Subsequent runs will load the index instantly.

Usage Enter any question about Dr. B.R. Ambedkar's speech.

Type quit, exit, or q to end the session.

### Example questions:

- Your question: What remedy does Dr. Ambedkar propose against caste? 
- Your question: What is said about social reform? 
- Your question: Who is the real enemy? 

### Functionality Breakdown 
- load_and_split_document(file_path="speech.txt") 

Purpose: Loads the specified text file and splits it into manageable, overlapping chunks.

How: Uses LangChain's document loader and text splitter.

Why: Chunks are required for embedding and retrieval.

- create_vector_store(chunks, persist\_directory) 

Purpose: Embeds the chunks using a HuggingFace model and stores them in a persistent ChromaDB directory.

How: Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.

Why: Creates a fast, locally accessible vector index for semantic search.

- create_qa_chain(vectorstore) 

Purpose: Sets up the RAG workflow, connecting Ollama LLM to ChromaDB retrieval.

How: Custom prompt instructs the LLM to answer only from retrieved context.

Why: Ensures question is answered factually (not hallucinated).

- main() 

Purpose: Orchestrates user interaction.

How: Checks for existing vector store, manages ingestion and retrieval, provides interactive command-line loop.

Why: User-friendly local Q&A system.

### Troubleshooting Ollama errors: 

Make sure ollama serve is running and the right model is installed.

"File not found": Ensure speech.txt exists in the same directory.

Slow first run: Initial embedding/indexing takes time, but future startup will be instant.

Corrupted venv warnings: See advice above about cleaning invalid distributions.

Project Structure 
 your\_project/ 
 ├── main.py 
 ├── requirements.txt 
 ├── speech.txt 
 ├── ChromaDB/ # Created automatically 
 └── README.md