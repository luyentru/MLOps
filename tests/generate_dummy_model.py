import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5


class MyEfficientNetModel(nn.Module):
    """
    A model leveraging EfficientNet as the base feature extractor.
    Matches the structure from the development branch.
    """

    def __init__(self, num_classes: int, pretrained: bool = False, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.base_model = efficientnet_b5(weights=None)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            # 1st layer
            nn.Linear(num_features, num_features // 2),
            nn.BatchNorm1d(num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # 2nd layer
            nn.Linear(num_features // 2, num_features // 4),
            nn.BatchNorm1d(num_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Output layer
            nn.Linear(num_features // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


def generate_dummy_model(num_classes: int, model_path: str):
    model = MyEfficientNetModel(num_classes=num_classes)
    state_dict = model.state_dict()

    for key, value in state_dict.items():
        if value.dtype in (torch.float32, torch.float64):
            state_dict[key] = torch.randn_like(value)
    model.load_state_dict(state_dict)

    print(f"Generated Dummy Model State Dict Keys: {list(state_dict.keys())}")
    torch.save(model.state_dict(), model_path)
    print(f"Dummy model saved to {model_path}")


if __name__ == "__main__":
    NUM_CLASSES = 4
    MODEL_PATH = "models/efficientnet.pth"
    generate_dummy_model(NUM_CLASSES, MODEL_PATH)
