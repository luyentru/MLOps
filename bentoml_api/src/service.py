import bentoml
from bentoml.io import Image, JSON
import numpy as np
from PIL import Image as PILImage
from pet_fac_rec.preprocessing import preprocess_image
from pet_fac_rec.data import label_mapping

model_runner = bentoml.models.get("efficientnet:latest").to_runner()

svc = bentoml.Service("efficientnet_service", runners=[model_runner])


@svc.api(input=Image(), output=JSON())
def predict(image: PILImage.Image):
    preprocessed_image = preprocess_image(image)
    logits = model_runner.run.run(preprocessed_image)
    probabilities = np.exp(logits.squeeze()) / np.sum(np.exp(logits.squeeze()))
    predicted_class = np.argmax(probabilities).item()
    probability = probabilities[predicted_class]

    return {
        "predicted_class": predicted_class,
        "class_name": list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class)],
        "probability": probability,
    }
