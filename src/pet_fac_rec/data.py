from pathlib import Path
import os
import pandas as pd
import kagglehub as kh
import shutil
from typing import Optional, Tuple
import typer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class MyDataset(Dataset):
    """Custom dataset for processing images and labels from a CSV file."""

    def __init__(self, csv_file: Path, transform: Optional[transforms.Compose] = None) -> None:
        """
        Initialize the dataset from a CSV file.

        Args:
            csv_file (Path): Path to the CSV file containing file paths and labels.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """Return a specific sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[Image.Image, int]: Transformed image and its label.
        """
        row = self.data.iloc[index]
        image_path = Path(row["file_path"])
        label = int(row["label"])

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def preprocess(output_folder: Path) -> None:
    """
    Preprocess the raw data and save it to the output folder.
    Downloads the dataset if not already present and generates a CSV file.
    """
    print("Preprocessing data...")

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Download dataset if not already present
    if not any(output_folder.iterdir()):
        print("Downloading dataset...")
        download_path = Path(kh.dataset_download("anshtanwar/pets-facial-expression-dataset"))
        print("Path to dataset files:", download_path)

        master_folder_path = download_path / "Master Folder"

        if master_folder_path.exists():
            # Move each file/subfolder in the Master_folder to the output directory
            for item in master_folder_path.iterdir():
                shutil.move(str(item), str(output_folder))
            print(f"Contents of 'Master Folder' moved to: {output_folder}")
        else:
            print("Master Folder not found in the dataset.")

    # Walk through each subdirectory and file to create a data list
    data = []
    label_mapping = {"happy": 0, "sad": 1, "angry": 2, "other": 3}

    for split in ["train", "valid", "test"]:
        split_path = output_folder / split
        for label_dir in split_path.iterdir():
            if label_dir.is_dir():
                for file in label_dir.iterdir():
                    if file.is_file():
                        # Map the label and store file path and label
                        data.append(
                            {
                                "split": split,
                                "label": label_mapping.get(label_dir.name.lower(), -1),
                                "file_path": str(file),
                            }
                        )

    # Convert the list to a DataFrame and save it as a CSV
    data_df = pd.DataFrame(data)
    csv_path = output_folder / "data.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"Data saved to CSV at: {csv_path}")


def get_default_transforms() -> transforms.Compose:
    """Return default transformations for the dataset."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


if __name__ == "__main__":
    typer.run(preprocess)
