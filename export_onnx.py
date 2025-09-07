# export_onnx.py
import torch
from load_model import wrapper, tokenizer  # import from previous file

# Dummy input
sentences = ["This is a test sentence"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    wrapper,
    (inputs["input_ids"], inputs["attention_mask"]),
    "miniv2.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size"}
    }
)

print("✅ Exported ONNX model as miniv2.onnx")
