import os
import shutil
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables.")
else:
    print(f"Loaded GROQ_API_KEY: {GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:]}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Constants
STUDY_FOLDER = "Study"
CHROMA_PATH = "chroma_db"

# Initialize Embeddings
# We use a model that runs locally
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

def get_vectorstore():
    """Returns the persistent vector store instance."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def get_loader(file_path):
    """Factory to get the appropriate loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    return None

@app.route('/ingest', methods=['POST'])
def ingest_documents():
    """Scans Study/ folder and indexes all documents."""
    if not os.path.exists(STUDY_FOLDER):
        return jsonify({"error": "Study folder not found"}), 404

    # 1. Identify files
    files = [f for f in os.listdir(STUDY_FOLDER) if os.path.isfile(os.path.join(STUDY_FOLDER, f))]
    supported_files = [f for f in files if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    if not supported_files:
        return jsonify({"message": "No supported documents found in Study folder"}), 200

    documents = []
    failed_files = []

    # 2. Load documents
    for file in supported_files:
        file_path = os.path.join(STUDY_FOLDER, file)
        loader = get_loader(file_path)
        if loader:
            try:
                docs = loader.load()
                # Add metadata for source identification if not present
                for doc in docs:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = file
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                failed_files.append(file)
    
    if not documents:
        return jsonify({"message": "Failed to load any documents", "failures": failed_files}), 500

    # 3. Split text
    # Overlapping chunks as requested
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 4. Store in Chroma
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Error clearing Chroma DB: {e}")
            # If we can't delete it (e.g. file in use), we might append. 
            # But duplicate content is bad for RAG. 
            # We'll try to proceed. 

    # Re-initialize vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
   
    return jsonify({
        "message": "Ingestion complete",
        "files_processed": len(supported_files),
        "chunks_created": len(splits),
        "failures": failed_files
    })

@app.route('/query', methods=['POST'])
def query_documents():
    """Answers a question using RAG."""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']

    # Check if vector db exists
    if not os.path.exists(CHROMA_PATH):
         return jsonify({"error": "Knowledge base is empty. Please run /ingest first."}), 400

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    
    # Define Prompt
    system_prompt = (
        "You are an AI study assistant. Answer the student's question based strictly on the following context. "
        "Do not hallucinate or use external knowledge. "
        "If the answer is not in the context, say 'The requested information is not available in the provided study materials.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke
    try:
        response = rag_chain.invoke({"input": question})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    answer = response["answer"]
    
    # Prepare sources and retrieved text
    sources = []
    retrieved_contexts = []
    seen_sources = set()
    
    for doc in response["context"]:
        # Extract source file name
        source_path = doc.metadata.get("source", "Unknown")
        file_name = os.path.basename(source_path)
        
        if file_name not in seen_sources:
            sources.append(file_name)
            seen_sources.add(file_name)
        
        retrieved_contexts.append(doc.page_content)

    return jsonify({
        "generated_answer": answer,
        "source_documents": sources,
        "retrieved_text_snippets": retrieved_contexts
    })

if __name__ == '__main__':
    # Running on local machine
    app.run(host='127.0.0.1', port=5000, debug=True)
