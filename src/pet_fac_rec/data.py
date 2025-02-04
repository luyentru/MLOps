import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Tuple

import kagglehub as kh
import pandas as pd
import torch
import typer
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pet_fac_rec.preprocessing import get_transforms
from pet_fac_rec.utils import show_image_and_target


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("reports/logs", exist_ok=True)
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)

# define image classes
label_mapping = {"happy": 0, "sad": 1, "angry": 2, "other": 3}


class MyDataset(Dataset):
    """Custom dataset for processing images and labels from a CSV file."""

    def __init__(self, csv_file: Path, split: str, transform: Optional[transforms.Compose] = None) -> None:
        """
        Initialize the dataset from a CSV file.

        Args:
            csv_file (Path): Path to the CSV file containing file paths and labels.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split]
        self.labels = self.data["label"].unique()
        self.name = split
        self.images = list(self.data["file_path"])
        self.target = torch.tensor(self.data["label"].values, dtype=torch.long)
        self.num_classes = len(self.labels)
        self.transform = transform or get_transforms()

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
    log.info("Preprocessing data...")

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if the dataset has already been moved
    if any((output_folder / split).exists() for split in ["train", "valid", "test"]):
        log.info("Dataset already exists. Skipping download.")
        return

    log.info("Downloading dataset...")
    download_path = Path(kh.dataset_download("anshtanwar/pets-facial-expression-dataset"))
    log.info(f"Dataset downloaded to: {download_path}")

    if not download_path.exists():
        log.info("Error: Download path does not exist. Dataset download failed.")
        return

    # Check the contents of the downloaded path
    log.info(f"Contents of the downloaded dataset: {[item.name for item in download_path.iterdir()]}")

    master_folder_path = download_path / "Master Folder"

    if master_folder_path.exists():
        log.info(f"Contents of 'Master Folder': {[item.name for item in master_folder_path.iterdir()]}")

        # Move the subdirectories (train, valid, test) from "Master Folder" to the output folder
        for subfolder in ["train", "valid", "test"]:
            source = master_folder_path / subfolder
            destination = output_folder / subfolder
            if source.exists():
                shutil.move(str(source), str(destination))
                log.info(f"Moved '{subfolder}' to {output_folder}")
            else:
                log.info(f"Warning: '{subfolder}' not found in 'Master Folder'.")
    else:
        log.info("Error: 'Master Folder' not found in the downloaded dataset.")
        return

    # Walk through each subdirectory and file to create a data list
    data = []

    for split in ["train", "valid", "test"]:
        split_path = output_folder / split
        if not split_path.exists():
            log.info(f"Warning: Split folder '{split}' not found in {output_folder}")
            continue
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
    log.info(f"Data saved to CSV at: {csv_path}")


def dataset_statistics(datadir: str = "data/data.csv") -> None:
    """Compute dataset statistics."""
    train_dataset = MyDataset(csv_file=Path(datadir), split="train")
    test_dataset = MyDataset(csv_file=Path(datadir), split="test")
    log.info(f"Train dataset: {train_dataset.name}")
    log.info(f"Number of images: {len(train_dataset)}")
    log.info(f"Image shape: {train_dataset[0][0].shape}")
    log.info("\n")
    log.info(f"Test dataset: {test_dataset.name}")
    log.info(f"Number of images: {len(test_dataset)}")
    log.info(f"Image shape: {test_dataset[0][0].shape}")

    show_image_and_target(train_dataset.images[:25], train_dataset.target[:25], show=False)
    plt.savefig("pet_facial_expressions.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


def main():
    typer.run(preprocess)


if __name__ == "__main__":
    main()
