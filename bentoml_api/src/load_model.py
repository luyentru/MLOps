import onnx
import bentoml

# Loading ONNX file
model_path = "models/efficientnet_v2.onnx"
model_proto = onnx.load(model_path)
onnx.checker.check_model(model_proto)
print("Model loaded succesfully!")

# Saving model into BentoML store
bento_model = bentoml.onnx.save_model(
    "efficientnet",
    model_proto,
    signatures={
        "run": {
            "batchable": True,
        }
    },
    metadata={
        "framework": "onnx",
        "description": "EfficientNetV2 model for image classification",
        "version": "1.0",
    },
)

print(f"Model saved: {bento_model}")
