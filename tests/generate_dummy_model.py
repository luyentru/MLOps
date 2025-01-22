import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5


class MyEfficientNetModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        self.base_model = efficientnet_b5(weights=None)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


def generate_dummy_model(num_classes: int, model_path: str):
    model = MyEfficientNetModel(num_classes=num_classes)
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        if value.dtype in (torch.float32, torch.float64):
            state_dict[key] = torch.randn_like(value)
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), model_path)
    print(f"Dummy model saved to {model_path}")


if __name__ == "__main__":
    NUM_CLASSES = 4
    MODEL_PATH = "models/efficientnet.pth"
    generate_dummy_model(NUM_CLASSES, MODEL_PATH)
