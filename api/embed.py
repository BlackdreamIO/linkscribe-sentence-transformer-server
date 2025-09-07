# api/embed.py
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import os

# Load ONNX model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "miniv2-quant.onnx")
session = ort.InferenceSession(MODEL_PATH)

# Load tokenizer
TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pre_tokenizer = Whitespace()

# Helper functions
def encode_sentences(sentences):
    encodings = [tokenizer.encode(s) for s in sentences]
    max_len = max(len(e.ids) for e in encodings)
    input_ids, attention_mask = [], []
    for e in encodings:
        ids = e.ids + [0] * (max_len - len(e.ids))
        mask = [1] * len(e.ids) + [0] * (max_len - len(e.ids))
        input_ids.append(ids)
        attention_mask.append(mask)
    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64)
    }

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

# FastAPI app
app = FastAPI()

class EmbedRequest(BaseModel):
    sentences: list[str]

@app.post("/embed")
def embed(request: EmbedRequest):
    onnx_inputs = encode_sentences(request.sentences)
    outputs = session.run(None, onnx_inputs)[0]
    embeddings = normalize(outputs).tolist()
    return {"embeddings": embeddings}
