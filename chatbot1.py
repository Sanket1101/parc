import os
import shutil
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

# Storage path for uploaded PDFs
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load optimized embeddings model (Smaller, Faster)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")  # Faster than MiniLM

# Define global variables for vector store & retriever
vector_store = None
qa_chain = None


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file and process it for question-answering."""
    global vector_store, qa_chain

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process all PDFs in the directory
    documents = []
    for pdf in os.listdir(UPLOAD_FOLDER):
        loader = PyPDFLoader(os.path.join(UPLOAD_FOLDER, pdf))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store for fast retrieval
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    retriever = vector_store.as_retriever()

    # Load a lightweight, quantized LLM for fast inference
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-Q4_K_M.gguf",  # Faster, quantized
        model_type="mistral",
        config={"stream": True, "max_new_tokens": 100, "temperature": 0.3}  # Faster & accurate
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return {"message": "File uploaded and processed successfully", "file_name": file.filename}


@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    """Ask a question based on the uploaded PDFs and stream response."""
    global qa_chain

    # Check if any PDFs exist in the directory
    if not os.listdir(UPLOAD_FOLDER):
        return {"error": "No PDFs uploaded. Please upload a file first."}

    # Ensure the model is initialized
    if qa_chain is None:
        return {"error": "PDFs exist but have not been processed yet. Try uploading again."}

    # Function to stream response
    def response_stream():
        for chunk in qa_chain.run(query):
            yield chunk  # Send response token by token

    return StreamingResponse(response_stream(), media_type="text/plain")


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
