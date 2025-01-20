import bentoml
from bentoml.io import Image, JSON
import numpy as np
from PIL import Image as PILImage

model_runner = bentoml.models.get("efficientnet:latest").to_runner()

svc = bentoml.Service("efficientnet_service", runners=[model_runner])


def preprocess_image(image: PILImage.Image) -> np.ndarray:
    # Resize image to the model's expected input size
    input_size = (224, 224)  # Example size; adjust to match your model
    image = image.resize(input_size).convert("RGB")

    # Convert image to numpy array
    image_array = np.array(image).astype(np.float32)

    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0

    # Add batch dimension (N, C, H, W)
    image_array = np.expand_dims(image_array.transpose(2, 0, 1), axis=0)
    return image_array


@svc.api(input=Image(), output=JSON())
def predict(image: PILImage.Image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Run inference
    predictions = model_runner.run.run(preprocessed_image)

    # Get the predicted class (assuming the model outputs probabilities)
    predicted_class = np.argmax(predictions, axis=1).item()

    return {"predicted_class": predicted_class}
