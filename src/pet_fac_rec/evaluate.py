import torch
import typer
from pathlib import Path
from pet_fac_rec.model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from pet_fac_rec.data import MyDataset, get_default_transforms

app = typer.Typer()


def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Returns the model based on the model name.
    """
    if model_name == "efficientnet":
        return MyEfficientNetModel(num_classes=num_classes)
    elif model_name == "resnet50":
        return MyResNet50Model(num_classes=num_classes)
    elif model_name == "vgg16":
        return MyVGG16Model(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type provided!")


@app.command()
def evaluate(
    model_name: str = typer.Option("efficientnet", help="Model type to use ('efficientnet', 'resnet50', 'vgg16')"),
    model_checkpoint: str = typer.Option(..., help="Path to the model checkpoint file (e.g., efficientnet.pth)"),
    data_csv: Path = Path("data/data.csv"),
) -> None:
    """
    Evaluate a trained model.
    """
    print("Evaluating...")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {model_checkpoint}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load the dataset
    test_set = MyDataset(csv_file=data_csv, split="test", transform=get_default_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    # Initialize the model
    num_classes = test_set.num_classes
    model = get_model(model_name, num_classes).to(device)

    # Load model checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    for img, target in test_dataloader:
        img, target = img.to(device), target.to(device)
        y_pred = model(img)
        _, preds = torch.max(y_pred, 1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    print(f"Test accuracy: {correct / total:.5f}")


if __name__ == "__main__":
    app()
