from fastapi.testclient import TestClient
from pet_fac_rec.api import app
from pathlib import Path

# Path to the test image directory (adjust as needed)
TEST_IMAGE_DIR = Path("data/valid/happy")

def test_predict_endpoint():
    """
    Test the /predict/ endpoint by sending a test image.
    """
    # Get the first image in the test directory
    test_image_path = next(TEST_IMAGE_DIR.glob("*.jpg"), None)  # Assumes JPEG files

    assert test_image_path is not None, "No test images found in the specified directory."

    # Use lifespan to ensure startup events run
    with TestClient(app) as client:
        # Open the test image in binary mode
        with open(test_image_path, "rb") as image_file:
            files = {"file": image_file}
            response = client.post("/predict/", files=files)

        # Validate the response
        assert response.status_code == 200
        json_response = response.json()
        assert isinstance(json_response, list)  # The response should be a list
        assert len(json_response) == 1          # The list should contain one prediction
        assert json_response[0] in ["happy", "sad", "angry", "other"]  # Check valid labels
