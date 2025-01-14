import torch
import typer
from pathlib import Path
import logging
from datetime import datetime
from pytorch_lightning import Trainer, LightningModule
from pet_fac_rec.model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from pet_fac_rec.data import MyDataset, get_default_transforms

app = typer.Typer()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)


def get_model_class(model_name: str) -> type:
    """
    Returns the model class based on the model name.
    """
    if model_name == "efficientnet":
        return MyEfficientNetModel
    elif model_name == "resnet50":
        return MyResNet50Model
    elif model_name == "vgg16":
        return MyVGG16Model
    else:
        raise ValueError("Unsupported model type provided!")


@app.command()
def evaluate(
    model_name: str = typer.Option("efficientnet", help="Model type to use ('efficientnet', 'resnet50', 'vgg16')"),
    model_checkpoint: str = typer.Option(..., help="Path to the model checkpoint file (e.g., efficientnet.ckpt)"),
    data_csv: Path = Path("data/data.csv"),
) -> None:
    """
    Evaluate a trained model.
    """
    log.info("Evaluating...")
    log.info(f"Model: {model_name}")
    log.info(f"Checkpoint: {model_checkpoint}")

    # Determine the device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load the dataset
    test_set = MyDataset(csv_file=data_csv, split="test", transform=get_default_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    # Get the model class
    model_class = get_model_class(model_name)

    # Load the checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path=model_checkpoint).to(device)

    # Use PyTorch Lightning's Trainer for evaluation
    trainer = Trainer(accelerator="auto", devices=1)
    trainer.test(model, dataloaders=test_dataloader)

    log.info("Evaluation complete")


if __name__ == "__main__":
    app()
