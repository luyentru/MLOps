import numpy as np
from PIL import Image
from torchvision import transforms


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    transform = get_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0).numpy()
