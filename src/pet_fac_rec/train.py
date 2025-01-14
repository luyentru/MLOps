import logging
from typing import List, Optional
import random
from datetime import datetime

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import typer
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
import torch
from pytorch_lightning.profilers import SimpleProfiler
from pet_fac_rec.model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from pet_fac_rec.data import MyDataset, get_default_transforms

app = typer.Typer()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)


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


def get_model(model_name: str, num_classes: int, learning_rate: float) -> LightningModule:
    """
    Returns the model based on the model name.
    """
    if model_name == "efficientnet":
        return MyEfficientNetModel(num_classes=num_classes, learning_rate=learning_rate)
    elif model_name == "resnet50":
        return MyResNet50Model(num_classes=num_classes, learning_rate=learning_rate)
    elif model_name == "vgg16":
        return MyVGG16Model(num_classes=num_classes, learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported model type provided!")


@app.command()
def train(
    model_name: str = typer.Option("efficientnet", help="Model type to use ('efficientnet', 'resnet50', 'vgg16')"),
    overrides: Optional[List[str]] = typer.Argument(None),
) -> None:
    cfg = my_compose(overrides)
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")  # Remove later
    log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiment

    # Set the seed for reproducibility
    set_seed(hparams.seed)

    # Load the dataset
    transform = get_default_transforms()
    train_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="train", transform=transform)
    valid_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="valid", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=False)

    # Get the model using the existing utility function
    num_classes = train_dataset.num_classes
    model = get_model(model_name, num_classes, hparams.lr)

    # Set up the SimpleProfiler
    profiler = SimpleProfiler()

    # Use PyTorch Lightning Trainer with SimpleProfiler
    trainer = Trainer(
        max_epochs=hparams.epochs,
        accelerator="auto",
        devices=1,
        profiler=profiler,  # Attach the SimpleProfiler
    )
    log.info(f"Starting training for {model_name}...")
    trainer.fit(model, train_dataloader, val_dataloader)
    log.info("Training complete")

    # Save the model
    model_save_path = f"models/{model_name}.ckpt"
    trainer.save_checkpoint(model_save_path)
    log.info(f"Model checkpoint saved to {model_save_path}")


if __name__ == "__main__":
    app()
