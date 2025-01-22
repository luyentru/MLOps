from pathlib import Path

from fastapi.testclient import TestClient

from pet_fac_rec.api import app


# Path to the test image file
TEST_IMAGE_PATH = Path("tests/testimage_happy.jpg")


def test_predict_endpoint():
    """
    Test the /predict/ endpoint by sending a test image.
    """
    # Ensure the test image exists
    assert TEST_IMAGE_PATH.exists(), f"Test image not found at {TEST_IMAGE_PATH}"

    # Use lifespan to ensure startup events run
    with TestClient(app) as client:
        # Open the test image in binary mode
        with open(TEST_IMAGE_PATH, "rb") as image_file:
            files = {"file": image_file}
            response = client.post("/predict/", files=files)

        # Validate the response
        assert response.status_code == 200
        json_response = response.json()
        assert isinstance(json_response, list)  # The response should be a list
        assert len(json_response) == 1  # The list should contain one prediction
        assert json_response[0] in ["happy", "sad", "angry", "other"]  # Check valid labels
