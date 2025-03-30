import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Define preprocessing steps
def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

# Preprocess all images in the dataset folder
def preprocess_dataset(dataset_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        processed_img = preprocess_image(img_path)
        torch.save(processed_img, os.path.join(save_path, img_name + '.pt'))

if __name__ == '__main__':
    dataset_path = "data/dataset"
    save_path = "data/processed"
    preprocess_dataset(dataset_path, save_path)
