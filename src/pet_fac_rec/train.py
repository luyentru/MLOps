import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import onnx
import torch
import typer
from dotenv import load_dotenv
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.profiler import record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from pet_fac_rec.data import MyDataset
from pet_fac_rec.model import MyEfficientNetModel
from pet_fac_rec.model import MyResNet50Model
from pet_fac_rec.model import MyVGG16Model
from pet_fac_rec.preprocessing import get_transforms
from pet_fac_rec.visualize import plot_training_statistics


app = typer.Typer()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_ENTITY_ORG = os.getenv("WANDB_ENTITY_ORG")
WANDB_REGISTRY = os.getenv("WANDB_REGISTRY")
WANDB_COLLECTION = os.getenv("WANDB_COLLECTION")

wandb.login(key=WANDB_API_KEY)


def my_compose(overrides: Optional[List[str]]) -> DictConfig:
    with initialize(config_path="configs", job_name="train_model"):
        return compose(config_name="config.yaml", overrides=overrides)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, num_classes: int, dropout: float) -> torch.nn.Module:
    """
    Returns the model based on the model name.
    """
    if model_name == "efficientnet":
        return MyEfficientNetModel(num_classes=num_classes, dropout_rate=dropout)
    elif model_name == "resnet50":
        return MyResNet50Model(num_classes=num_classes)
    elif model_name == "vgg16":
        return MyVGG16Model(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type provided!")


def save_and_log_artifact(
    wandb: wandb,
    run: wandb.sdk.wandb_run.Run,
    model_save_path: str,
    train_losses: List[float],
    train_accuracies: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
) -> None:
    """
    Save and log the trained model as an artifact in Weights & Biases (wandb).

    Args:
        wandb (wandb): The wandb module.
        run (wandb.sdk.wandb_run.Run): The current wandb run.
        model_save_path (str): Path to the saved model file.
        train_losses (List[float]): List of training losses.
        train_accuracies (List[float]): List of training accuracies.
        val_losses (List[float]): List of validation losses.
        val_accuracies (List[float]): List of validation accuracies.
    """
    # Create a new artifact for the model
    artifact = wandb.Artifact(
        name=WANDB_COLLECTION,
        type="model",
        description="A model trained to classify facial expressions of animals",
        metadata={
            "train_loss": train_losses[-1],
            "train_accuracy": train_accuracies[-1],
            "valid_loss": val_losses[-1],
            "valid_accuracy": val_accuracies[-1],
        },
    )

    # Add the model file to the artifact
    artifact.add_file(model_save_path)

    # Log the artifact to the current run
    logged_art = run.log_artifact(artifact)

    # Link the artifact to a specific path in the wandb registry
    run.link_artifact(
        artifact=logged_art,
        target_path=f"{WANDB_ENTITY_ORG}/{WANDB_REGISTRY}/{WANDB_COLLECTION}",
        aliases=["staging"],
    )
    artifact.save()


def save_onnx_file(model_name: str, model: torch.nn.Module, device: torch.device) -> None:
    """
    Export a PyTorch model to ONNX format and save it to a file.

    Args:
        model_name (str): The name of the model.
        model (torch.nn.Module): The PyTorch model to be exported.
        device (torch.device): The device on which the model is located.
    """
    # Export ONNX file
    try:
        # Set the model to evaluation mode
        model.eval()

        # Create a dummy input tensor with the appropriate shape
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        # Define the path to save the ONNX model
        onnx_save_path = f"models/{model_name}.onnx"

        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,  # Specify ONNX opset version
            do_constant_folding=True,  # Optimize constant-folding
            export_params=True,  # Store the trained parameter weights inside the model file
        )

        # Verify the model
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)

        # Add model metadata
        onnx_model.graph.doc_string = f"Pet facial recognition model using {model_name}"
        onnx.save(onnx_model, onnx_save_path)

        log.info(f"Model exported and verified to ONNX at {onnx_save_path}")
    except Exception as e:
        log.error(f"Failed to export ONNX model: {str(e)}")


@app.command()
def train(
    model_name: str = typer.Option("efficientnet", help="Model type to use ('efficientnet', 'resnet50', 'vgg16')"),
    overrides: Optional[List[str]] = typer.Argument(None),
) -> None:
    """
    Train a model with the specified configuration.

    Args:
        model_name (str): The type of model to use ('efficientnet', 'resnet50', 'vgg16').
        overrides (Optional[List[str]]): A list of configuration overrides.
    """
    cfg = my_compose(overrides)
    log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiment

    # Set the seed for reproducibility
    set_seed(hparams.seed)

    # Determine the device to use for training
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info(f"Running on dev: {device}")  # Remove later

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="train",
        name=f"exp_{current_time}",
        config={
            "lr": hparams.lr,
            "batch_size": hparams.batch_size,
            "epochs": hparams.epochs,
            "dropout": hparams.dropout,
        },
    )

    # Load the dataset
    transform = get_transforms()
    train_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="train", transform=transform)
    valid_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="valid", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=True)

    # Initialize the model
    num_classes = train_dataset.num_classes
    model = get_model(model_name, num_classes, hparams.dropout).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    # Training loop
    log.info(f"Start training {model_name}...")
    epoch_bar = tqdm(range(hparams.epochs))
    for epoch in epoch_bar:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler"),
            with_stack=True,
        ) as profiler:
            model.train()
            train_loss = 0.0
            total_correct = 0
            total_samples = 0

            for img, target in iter(train_dataloader):
                img, target = img.to(device), target.to(device)

                # Forward pass
                y_pred = model(img)
                with record_function("model_loss"):
                    loss = criterion(y_pred, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                with record_function("backward"):
                    loss.backward()
                    optimizer.step()
                profiler.step()

                train_loss += loss.item() * img.size(0)
                _, preds = torch.max(y_pred, 1)
                total_correct += (preds == target).sum().item()
                total_samples += target.size(0)

            train_losses.append(train_loss / total_samples)
            train_accuracies.append(total_correct / total_samples)

            val_loss = 0.0
            total_correct = 0
            total_samples = 0

            model.eval()

            with torch.no_grad():
                for img, target in iter(val_dataloader):
                    img, target = img.to(device), target.to(device)

                    outputs = model(img)
                    loss = criterion(outputs, target)

                    val_loss += loss.item() * img.size(0)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == target).sum().item()
                    total_samples += target.size(0)

            val_losses.append(val_loss / total_samples)
            val_accuracies.append(total_correct / total_samples)
            status = (
                f"Epoch {epoch + 1}/{hparams.epochs} | "
                f"Train Loss: {train_losses[-1]:.5f} | "
                f"Train Acc: {train_accuracies[-1]:.5f} | "
                f"Val Loss: {val_losses[-1]:.5f} | "
                f"Val Acc: {val_accuracies[-1]:.5f}"
            )
            epoch_bar.set_description(status)
            log.info(status)
            wandb.log(
                {
                    "train_loss": train_losses[-1],
                    "train_accuracy": train_accuracies[-1],
                    "valid_loss": val_losses[-1],
                    "valid_accuracy": val_accuracies[-1],
                }
            )

    log.info("Training complete")

    # Save the model
    model_save_path = f"models/{model_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    log.info(f"Model saved to {model_save_path}")

    save_and_log_artifact(wandb, run, model_save_path, train_losses, train_accuracies, val_losses, val_accuracies)
    save_onnx_file(model_name, model, device)

    # Plot training statistics
    plot_training_statistics(train_losses, train_accuracies, val_losses, val_accuracies)

    wandb.finish()


if __name__ == "__main__":
    app()
