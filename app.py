import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

# --------- Load model and tokenizer ONCE ---------
MODEL_PATH = "miniv2-quant.onnx"
tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer.pre_tokenizer = Whitespace()

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"]
)

# --------- FastAPI setup ---------
app = FastAPI(title="MiniLMv2 ONNX Embedding Server")

class EmbedRequest(BaseModel):
    sentences: list[str]

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def encode_sentences(sentences: list[str]):
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

# --------- API endpoint ---------
@app.post("/embed")
def embed(request: EmbedRequest):
    onnx_inputs = encode_sentences(request.sentences)
    outputs = session.run(None, onnx_inputs)[0]
    embeddings = normalize(outputs).tolist()
    return {"embeddings": embeddings}
