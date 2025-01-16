import wandb
import os
import time
import torch
from pet_fac_rec.model import MyEfficientNetModel
from dotenv import load_dotenv

load_dotenv()


def test_model_speed():
    model_name = os.getenv("MODEL_NAME")
    run = wandb.init()
    artifact = run.use_artifact(
        "luyentrungkien00-danmarks-tekniske-universitet-dtu-org/wandb-registry-model/pet-fac-rec-model:v0", type="model"
    )
    artifact_dir = artifact.download()
    model = MyEfficientNetModel(4)
    model.load_state_dict(torch.load(f"{artifact_dir}/{model_name}.pth"))
    model.eval()

    # Generate a random input appropriate for your model
    test_input = torch.rand(1, 3, 224, 224)  # Example for an image model

    # Timing the model predictions
    start_time = time.time()
    num_predictions = 1
    for _ in range(num_predictions):
        model(test_input)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {num_predictions} predictions: {total_time:.2f} seconds")

    # Assert that 100 predictions are made in less than X seconds, adjust X as needed
    assert total_time < 10, "Model predictions are too slow"


if __name__ == "__main__":
    test_model_speed()
