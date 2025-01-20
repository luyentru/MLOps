import bentoml
from bentoml.io import Image, JSON
import numpy as np
from PIL import Image as PILImage
from pet_fac_rec.preprocessing import preprocess_image

model_runner = bentoml.models.get("efficientnet:latest").to_runner()

svc = bentoml.Service("efficientnet_service", runners=[model_runner])


@svc.api(input=Image(), output=JSON())
def predict(image: PILImage.Image):
    preprocessed_image = preprocess_image(image)
    predictions = model_runner.run.run(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1).item()
    return {"predicted_class": predicted_class}
