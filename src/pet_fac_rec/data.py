from pathlib import Path
import os
import pandas as pd
import shutil
import kagglehub as kh
import typer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, csv_file: Path, transform=None) -> None:
        """
        Initialize the dataset from a CSV file.

        Args:
            csv_file (Path): Path to the CSV file containing file paths and labels.
            transform: Transformations to apply to the images.
        """
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Transformed image and its label.
        """
        row = self.data.iloc[index]
        image_path = row["file_path"]
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
        download_path = kh.dataset_download("anshtanwar/pets-facial-expression-dataset")
        print("Path to dataset files:", download_path)

        # Move dataset to the target directory
        shutil.move(download_path, output_folder)
        print(f"Dataset moved to: {output_folder}")

    # Define the root directory of the dataset
    root_dir = output_folder / "11/Master Folder"

    # Initialize an empty list to store data
    data = []

    # Define label mapping
    label_mapping = {"happy": 0, "Sad": 1, "Angry": 2, "Other": 3} 

    # Walk through each subdirectory and file
    for split in ["train", "valid", "test"]:
        split_path = root_dir / split
        for label in os.listdir(split_path):  # Angry, Happy, Sad, etc.
            label_path = split_path / label
            if label_path.is_dir():
                for file in os.listdir(label_path):
                    file_path = label_path / file
                    if os.path.isfile(file_path):
                        # Append file path and label to the data list
                        if label in label_mapping:
                            data.append({
                                "split": split,
                                "label": label_mapping[label],  # Convert string label to integer
                                "file_path": str(file_path)
                            })
                        else:
                            print(f"Warning: Label '{label}' not found in label mapping")

    # Convert the list to a pandas DataFrame
    data_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_path = output_folder / "data.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"Data saved to CSV at: {csv_path}")

# Define default transformations for the dataset
def get_default_transforms():
    """Return default transformations for the dataset."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == "__main__":
    typer.run(preprocess)
