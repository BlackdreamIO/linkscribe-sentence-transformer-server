from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Initialize FastAPI
app = FastAPI(title="MiniLM Embedding API")

# Load MiniLM L6 v2
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request body model
class TextRequest(BaseModel):
    text: str

# Endpoint to get embeddings
@app.post("/embed")
def embed_text(request: TextRequest):
    embedding = model.encode([request.text])[0].tolist()
    return {"embedding": embedding}

# Optional health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
