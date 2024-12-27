import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


app = FastAPI(title="Local RAG Server")

class QueryRequest(BaseModel):
    question: str

index = None
query_engine = None

@app.on_event("startup")
def setup():
    """
    This function is called once when the server starts.
    We'll:
      1) Configure the global LlamaIndex Settings with local embeddings & Ollama LLM
      2) Load the stored VectorStoreIndex from disk
      3) Create a query engine we can reuse
    """
    global index, query_engine

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)

    storage_context = StorageContext.from_defaults(persist_dir="documents/index")
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_query_engine()
    print("Startup complete: index loaded and query engine ready.")


@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Expects a JSON like:
    {
      "question": "What is Ollama?"
    }
    Returns:
    {
      "answer": "...some answer from the LLM..."
    }
    """
    response = query_engine.query(request.question)
    return {"answer": str(response)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)