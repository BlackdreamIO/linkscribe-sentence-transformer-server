# quantize_model.py
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_FP32 = "miniv2.onnx"
MODEL_INT8 = "miniv2-quant.onnx"

# Quantize weights to int8
quantize_dynamic(
    model_input=MODEL_FP32,
    model_output=MODEL_INT8,
    weight_type=QuantType.QUInt8
)

print(f"✅ Quantized ONNX model saved as {MODEL_INT8}")
