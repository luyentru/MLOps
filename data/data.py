import os
import pandas as pd
import shutil
import kagglehub as kh

# Download dataset
download_path = kh.dataset_download("anshtanwar/pets-facial-expression-dataset")
print("Path to dataset files:", download_path)

# Define target directory
current_directory = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.join(current_directory, "pets_facial_expression_dataset")

# Move dataset to the target directory
shutil.move(download_path, target_directory)

print(f"Dataset moved to: {target_directory}")

# Define the root directory of the dataset
root_dir = os.path.join(target_directory, "Master Folder")

# Initialize an empty list to store data
data = []

# Walk through each subdirectory and file
for split in ["train", "valid", "test"]:
    split_path = os.path.join(root_dir, split)
    for label in os.listdir(split_path):  # Angry, Happy, Sad, etc.
        label_path = os.path.join(split_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if os.path.isfile(file_path):
                    # Append file path and label to the data list
                    data.append({"split": split, "label": label, "file_path": file_path})

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file (optional)
csv_path = os.path.join(current_directory, "data.csv")
df.to_csv(csv_path, index=False)
