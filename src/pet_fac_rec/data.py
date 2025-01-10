from pathlib import Path
import os
import pandas as pd
import shutil
import kagglehub as kh
import typer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.data.iloc[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # Download dataset
        download_path = kh.dataset_download("anshtanwar/pets-facial-expression-dataset")
        print("Path to dataset files:", download_path)

        # Move dataset to the target directory
        shutil.move(download_path, output_folder)
        print(f"Dataset moved to: {output_folder}")

        # Define the root directory of the dataset
        root_dir = output_folder / "Master Folder"

        # Initialize an empty list to store data
        data = []

        # Walk through each subdirectory and file
        for split in ["train", "valid", "test"]:
            split_path = root_dir / split
            for label in os.listdir(split_path):  # Angry, Happy, Sad, etc.
                label_path = split_path / label
                if label_path.is_dir():
                    for file in os.listdir(label_path):
                        file_path = label_path / file
                        if file_path.is_file():
                            # Append file path and label to the data list
                            data.append({"split": split, "label": label, "file_path": str(file_path)})

        # Convert the list to a pandas DataFrame
        self.data = pd.DataFrame(data)

        # Save the DataFrame to a CSV file (optional)
        csv_path = output_folder / "data.csv"
        self.data.to_csv(csv_path, index=False)
        print(f"Data saved to CSV at: {csv_path}")

def preprocess(output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(output_folder)  # Use output_folder directly
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
