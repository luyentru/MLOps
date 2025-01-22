import bentoml
import numpy as np
from bentoml.io import JSON
from bentoml.io import Image
from PIL import Image as PILImage
from prometheus_client import Counter
from prometheus_client import Summary
from prometheus_client import make_asgi_app

from pet_fac_rec.data import label_mapping
from pet_fac_rec.preprocessing import preprocess_image


# Define Prometheus metrics
prediction_counter = Counter("prediction_requests_total", "Total number of predictions")
error_counter = Counter("prediction_error", "Number of prediction errors")
prediction_latency = Summary("prediction_latency_seconds", "Time spent processing prediction")


model_runner = bentoml.models.get("efficientnet:latest").to_runner()
svc = bentoml.Service("efficientnet_service", runners=[model_runner])

# Mount prometheus metrics endpoint
svc.mount_asgi_app(make_asgi_app(), path="/metrics")


@svc.api(input=Image(), output=JSON())
def predict(image: PILImage.Image):
    try:
        prediction_counter.inc()
        with prediction_latency.time():
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
    except Exception as e:
        error_counter.inc()
        raise e
