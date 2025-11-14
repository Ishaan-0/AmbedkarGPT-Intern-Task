from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_and_split_document(file_path="speech.txt"):
    """
    Load the text file and split it into manageable chunks.
    
    Args:
        file_path (str): Path to the speech text file
        
    Returns:
        list: List of document chunks
    """
    print("Loading document...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the document.")
    
    return chunks

def create_vector_store(chunks, persist_directory="../ChromaDB"):
    """
    Create embeddings and store them in ChromaDB.
    
    Args:
        chunks (list): Document chunks to embed
        persist_directory (str): Directory to persist the vector store
        
    Returns:
        Chroma: ChromaDB vector store instance
    """
    print("Creating embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("Vector store created successfully!")
    return vectorstore

def create_qa_chain(vectorstore):
    """
    Create a RetrievalQA chain using ChatOllama and the vector store.
    
    Args:
        vectorstore (Chroma): The vector store containing document embeddings
        
    Returns:
        RetrievalQA: The QA chain
    """
    print("Initializing Ollama LLM...")
    llm = ChatOllama(
        model="mistral:7b",
        temperature=0,  # Lower temperature for more consistent, factual responses
    )
    
    # Custom prompt template for better context-aware responses
    prompt_template = """You are an AI assistant answering questions based solely on the provided context from Dr. B.R. Ambedkar's speech.
    
Context: {context}

Question: {question}

Answer the question based only on the context provided above. If the answer cannot be found in the context, say "I cannot answer this question based on the provided text."

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    print("Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def main():
    """
    Main function to run the RAG-based Q&A system.
    """
    print("=" * 60)
    print("AmbedkarGPT - RAG-based Q&A System")
    print("=" * 60)
    
    # Check if vector store already exists
    persist_dir = "../ChromaDB"
    speech_file = "speech.txt"
    
    if not os.path.exists(speech_file):
        print(f"Error: {speech_file} not found!")
        print("Please create speech.txt with the provided text.")
        return
    
    if os.path.exists(persist_dir):
        print("Loading existing vector store...\n")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...\n")
        # Load and process the document
        chunks = load_and_split_document(speech_file)
        
        # Create vector store with embeddings
        vectorstore = create_vector_store(chunks, persist_dir)
    
    qa_chain = create_qa_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about Dr. Ambedkar's speech.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    # Q&A loop
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using AmbedkarGPT!")
            break
        
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        print("\nThinking...\n")
        
        try:
            result = qa_chain.invoke({"query": question})
            
            print("Answer:", result['result'])
            print("\n" + "-" * 60 + "\n")
        
        except Exception as e:
            print(f"Error: {e}")
            print("Please make sure Ollama is running with 'ollama serve'\n")

if __name__ == "__main__":
    main()
