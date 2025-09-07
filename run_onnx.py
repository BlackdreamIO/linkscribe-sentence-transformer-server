# run_onnx.py
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import json

MODEL_PATH = "miniv2-quant.onnx"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load ONNX and tokenizer
session = ort.InferenceSession(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Example sentences
sentences = [
    "The cat sits on the mat.",
    "A dog is playing in the park.",
    "MiniLMv2 embeddings are ready for similarity search."
]

# Tokenize input
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")
onnx_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}

# Run ONNX inference
outputs = session.run(None, onnx_inputs)[0]

# Normalize embeddings
embeddings = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)

# Print result
print("✅ Embeddings shape:", embeddings.shape)
print("✅ First vector snippet:", embeddings[0][:5])

# Optional: save embeddings to JSON
with open("embeddings.json", "w") as f:
    json.dump(embeddings.tolist(), f)
print("✅ Saved embeddings to embeddings.json")
