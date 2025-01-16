import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from .model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from typing import List

app = FastAPI()

# Global variables for model and device
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the label mapping
label_mapping = {0: "happy", 1: "sad", 2: "angry", 3: "other"}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse to map index to label

# Define transformations to apply to input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_name: str, model_path: str, num_classes: int) -> torch.nn.Module:
    """
    Load the specified model from a checkpoint.
    """
    if model_name == "efficientnet":
        model = MyEfficientNetModel(num_classes=num_classes)
    elif model_name == "resnet50":
        model = MyResNet50Model(num_classes=num_classes)
    elif model_name == "vgg16":
        model = MyVGG16Model(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


@app.on_event("startup")
async def startup_event():
    """
    Initialize the model at startup.
    """
    global model
    model_name = "efficientnet"  # Change this to "resnet50" or "vgg16" as needed
    model_path = f"models/{model_name}.pth"  # Path to the model checkpoint
    num_classes = 4  # Update based on your dataset
    model = load_model(model_name, model_path, num_classes)
    print(f"{model_name} model loaded for inference.")

@app.post("/predict/", response_model=List[str])
async def predict(file: UploadFile = File(...)) -> List[str]:
    """
    Perform inference on an uploaded image and return the predicted emotion.
    """
    global model

    # Load image
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)  # Get the predicted class index

    # Map prediction to emotion using the label mapping
    predicted_emotion = label_mapping.get(preds.item(), "unknown")

    # Return prediction as a list
    return [predicted_emotion]
