import torch
import typer
from pathlib import Path
from pet_fac_rec.model import MyEfficientNetModel
from pet_fac_rec.data import MyDataset, get_default_transforms


def evaluate(model_checkpoint: str, data_csv: Path = Path("data/data.csv")) -> None:
    """Evaluate a trained model."""
    print("Evaluating...")
    print(model_checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    test_set = MyDataset(csv_file=data_csv, split="test", transform=get_default_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    num_classes = test_set.num_classes
    model = MyEfficientNetModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_checkpoint))

    model.eval()
    correct = 0
    total = 0
    for img, target in test_dataloader:
        img, target = img.to(device), target.to(device)
        y_pred = model(img)
        _, preds = torch.max(y_pred, 1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


def main():
    typer.run(evaluate)


if __name__ == "__main__":
    main()
