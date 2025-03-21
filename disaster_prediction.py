# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import subprocess
from zipfile import ZipFile

# Install required dependencies
def install_dependencies():
    try:
        import kaggle
    except ImportError:
        subprocess.run(["pip", "install", "kaggle"])

install_dependencies()

# Set up Kaggle API credentials
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.system("cp kaggle.json ~/.kaggle/")
os.system("chmod 600 ~/.kaggle/kaggle.json")

# Download dataset using Kaggle API
os.system("kaggle datasets download -d varpit94/disaster-images-dataset")

# Extract dataset
dataset_zip = "disaster-images-dataset.zip"

if os.path.exists(dataset_zip):
    with ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall("dataset")
    print("Dataset extracted!")
else:
    print("Error: Dataset file not found!")

# Define dataset paths
DATASET_PATH = "dataset/Comprehensive Disaster Dataset(CDD)"
categories = {
    "earthquake": f"{DATASET_PATH}/Damaged_Infrastructure/Earthquake",
    "infrastructure": f"{DATASET_PATH}/Damaged_Infrastructure/Infrastructure",
    "urban_fire_disaster": f"{DATASET_PATH}/Fire_Disaster/Urban_Fire",
    "wild_fire_disaster": f"{DATASET_PATH}/Fire_Disaster/Wild_Fire",
    "human_damage": f"{DATASET_PATH}/Human_Damage",
    "drought": f"{DATASET_PATH}/Land_Disaster/Drought",
    "landslide": f"{DATASET_PATH}/Land_Disaster/Land_Slide",
    "water_disaster": f"{DATASET_PATH}/Water_Disaster"
}

# Check files in each category
for category, path in categories.items():
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"📂 {category}: {len(files)} images | Sample: {files[:5]}")
    else:
        print(f"⚠️ Warning: {category} directory not found!")

# Assign numerical labels to categories
labels_dict = {
    "earthquake": 1,
    "infrastructure": 2,
    "urban_fire_disaster": 3,
    "wild_fire_disaster": 4,
    "human_damage": 5,
    "drought": 6,
    "landslide": 7,
    "water_disaster": 8
}

# Image Preprocessing
data = []
labels = []

for category, path in categories.items():
    if not os.path.exists(path):
        continue
    
    category_label = labels_dict[category]
    
    for img_file in os.listdir(path)[:500]:  # Load up to 500 images per category
        img_path = os.path.join(path, img_file)
        
        try:
            image = Image.open(img_path)
            image = image.resize((128, 128)).convert("RGB")
            data.append(np.array(image))
            labels.append(category_label)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")

# Convert data to NumPy arrays
data = np.array(data)
labels = np.array(labels)

print(f"✅ Loaded {len(data)} images with labels")

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"📊 Train Size: {len(X_train)}, Test Size: {len(X_test)}")

# Save preprocessed dataset
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data preprocessing completed and saved as NumPy arrays!")

# Display sample images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i])
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.show()
