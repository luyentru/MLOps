import pytest
from pathlib import Path
import logging
import os
import pandas as pd


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logging.getLogger().setLevel(logging.INFO)


def test_dataset_download(tmp_path, caplog):
    def test_dataset_download(tmp_path, caplog):
        """Test dataset download and preprocessing functionality"""
        caplog.set_level(logging.INFO)

        # Create mock input and output paths
        test_download_path = tmp_path / "mock_download"
        output_folder = tmp_path / "output"

        # Create mock dataset structure
        master_folder = test_download_path / "Master Folder"
        master_folder.mkdir(parents=True)

        # Create split folders with mock image data
        for split in ["train", "valid", "test"]:
            # Verify data integrity
            assert os.path.getsize(output_folder / "data.csv") > 0, "CSV file is empty"

            # Verify split folder contents
            for split in ["train", "valid", "test"]:
                split_path = output_folder / split
                assert len(list(split_path.iterdir())) > 0, f"{split} folder is empty"

                # Verify label folders exist
                for label in ["happy", "sad", "angry", "other"]:
                    label_path = split_path / label
                    assert label_path.exists(), f"Label folder {label} missing in {split}"
                    assert len(list(label_path.iterdir())) > 0, f"No images found in {split}/{label}"

            # Verify CSV contents
            df = pd.read_csv(output_folder / "data.csv")
            assert "split" in df.columns, "CSV missing 'split' column"
            assert "label" in df.columns, "CSV missing 'label' column"
            assert "file_path" in df.columns, "CSV missing 'file_path' column"
            assert not df.empty, "CSV contains no data"

            # Verify label values
            labels = df["label"].unique()
            assert all(label in [0, 1, 2, 3] for label in labels), "Invalid label values found"

            # Verify file paths in CSV exist
            assert all(Path(fp).exists() for fp in df["file_path"]), "Some image files in CSV don't exist"

            # Verify logging of completion
            assert any("Data saved to CSV" in record.message for record in caplog.records), (
                "Missing CSV save confirmation log"
            )
