import pytest
from pathlib import Path
from unittest.mock import patch
import logging
from pet_fac_rec.data import preprocess
from . import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logging.getLogger().setLevel(logging.INFO)

def test_dataset_download(tmp_path, caplog):
    """Test 1:
    - functionality to download dataset from Kaggle
    - functionality to create sub-folder structure
    - functionality to create logging
    
    """
    caplog.set_level(logging.INFO)
    
    test_download_path = Path(_PATH_DATA) / "download"
    
    # Create mock dataset structure
    master_folder = test_download_path / "Master Folder"
    master_folder.mkdir(parents=True)
    for split in ["train", "valid", "test"]:
        (master_folder / split).mkdir()
    
    # Mock the kagglehub download function
    with patch('kagglehub.dataset_download') as mock_download:
        mock_download.return_value = str(test_download_path)
        
        # Run preprocess with a test output folder
        output_folder = tmp_path / "output"
        preprocess(output_folder)
        
        # Verify download was attempted
        mock_download.assert_called_once_with("anshtanwar/pets-facial-expression-dataset")
        
        # Verify directory structure
        assert (output_folder / "train").exists(), "train folder doesnt exist"
        assert (output_folder / "valid").exists(), "valid folder doesnt exist"
        assert (output_folder / "test").exists(), "test folder doesnt exist"
        
        # Verify logging messages
        assert any("Downloading dataset..." in record.message for record in caplog.records), "logs were not created successfully"
        assert any(str(test_download_path) in record.message for record in caplog.records), "logs were not created successfully"