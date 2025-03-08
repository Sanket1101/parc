import os
import shutil
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# Initialize FastAPI
app = FastAPI()

# Directories
UPLOAD_FOLDER = "uploaded_files"
FAISS_INDEX_PATH = "faiss_index"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Load optimized embeddings model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Define global variables for vector store & retriever
vector_store = None
qa_chain = None


def load_existing_faiss():
    """Load FAISS index from disk if available."""
    global vector_store, qa_chain

    faiss_file = os.path.join(FAISS_INDEX_PATH, "faiss_index.pkl")

    if os.path.exists(faiss_file):
        with open(faiss_file, "rb") as f:
            vector_store = pickle.load(f)
            print("✅ FAISS index loaded from disk.")

    if vector_store:
        retriever = vector_store.as_retriever()

        # Load a lightweight, quantized LLM
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-Q4_K_M.gguf",  # Optimized for CPU
            model_type="mistral",
            config={"stream": True, "max_new_tokens": 100, "temperature": 0.3}
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple PDFs and process them for question-answering."""
    global vector_store, qa_chain

    # Save uploaded files
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Load and process all PDFs
    documents = []
    for pdf in os.listdir(UPLOAD_FOLDER):
        loader = PyPDFLoader(os.path.join(UPLOAD_FOLDER, pdf))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # Save FAISS index to disk
    with open(os.path.join(FAISS_INDEX_PATH, "faiss_index.pkl"), "wb") as f:
        pickle.dump(vector_store, f)
    print("✅ FAISS index saved to disk.")

    retriever = vector_store.as_retriever()

    # Load optimized LLM
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-Q4_K_M.gguf",
        model_type="mistral",
        config={"stream": True, "max_new_tokens": 100, "temperature": 0.3}
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return {"message": f"{len(files)} files uploaded and processed successfully"}


@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    """Ask a question based on the uploaded PDFs and stream response."""
    global qa_chain

    # Ensure FAISS index is loaded
    if vector_store is None or qa_chain is None:
        return {"error": "No processed PDFs found. Upload files first."}

    # Function to stream response
    def response_stream():
        for chunk in qa_chain.run(query):
            yield chunk  # Stream response token by token

    return StreamingResponse(response_stream(), media_type="text/plain")


# Load FAISS index on startup
load_existing_faiss()

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
