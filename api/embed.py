from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# Load ONNX model once
#MODEL_PATH = "./miniv2-quant.onnx"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "miniv2-quant.onnx")
session = ort.InferenceSession(MODEL_PATH)

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

app = FastAPI()

class EmbedRequest(BaseModel):
    sentences: list[str]

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

@app.post("/embed")
def embed(request: EmbedRequest):
    inputs = tokenizer(request.sentences, padding=True, truncation=True, return_tensors="np")
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    outputs = session.run(None, onnx_inputs)[0]
    embeddings = normalize(outputs).tolist()
    return {"embeddings": embeddings}
