import os
import platform
import requests
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from mangum import Mangum

# ------------------------------
# 1️⃣ Define a temporary folder
# ------------------------------
if platform.system() == "Windows":
    tmp_dir = os.path.join(os.getcwd(), "tmp")  # local tmp folder
    os.makedirs(tmp_dir, exist_ok=True)
else:
    tmp_dir = "/tmp"

MODEL_PATH = os.path.join(tmp_dir, "miniv2-quant.onnx")

# ------------------------------
# 2️⃣ Download ONNX model if missing
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1OQdBp2IWx_FEpk7XjTsGpA9XKHltyi7-"

if not os.path.exists(MODEL_PATH):
    print("🔃 Downloading ONNX model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Download complete!")

# ------------------------------
# 3️⃣ Load ONNX Runtime
# ------------------------------
session = ort.InferenceSession(MODEL_PATH)

# ------------------------------
# 4️⃣ Load tokenizer
# ------------------------------
TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pre_tokenizer = Whitespace()

# ------------------------------
# 5️⃣ Helpers
# ------------------------------
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

# ------------------------------
# 6️⃣ FastAPI app
# ------------------------------
app = FastAPI()

class EmbedRequest(BaseModel):
    sentences: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(request: EmbedRequest):
    onnx_inputs = encode_sentences(request.sentences)
    outputs = session.run(None, onnx_inputs)[0]
    embeddings = normalize(outputs).tolist()
    return {"embeddings": embeddings}

handler = Mangum(app)