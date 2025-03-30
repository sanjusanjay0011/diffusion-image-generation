import torch
from models.diffusion_model import DiffusionModel
from models.denoising_diffusion import noise_image, denoise
from PIL import Image
import os
import numpy as np
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DiffusionModel().to(device)
model.load_state_dict(torch.load(config['model_checkpoint_path'] + "model_epoch_50.pth"))
model.eval()

def generate_image(image, model, circle_radius=80):
    # Apply noise to the center of the image (circle)
    image = image.clone()
    h, w = image.shape[1], image.shape[2]
    cx, cy = w // 2, h // 2
    mask = torch.ones_like(image)
    mask[:, cy-circle_radius:cy+circle_radius, cx-circle_radius:cx+circle_radius] = 0
    noisy_image = noise_image(image) * mask

    # Denoise the image using the model
    denoised_image = denoise(noisy_image, model)

    # Combine the original image and denoised image
    result_image = image * mask + denoised_image * (1 - mask)
    return result_image

def save_generated_images():
    os.makedirs(config['generated_images_path'], exist_ok=True)
    for i in range(10):  # Generate 10 images
        img = torch.randn(1, 3, 256, 256).to(device)  # Random noise image
        generated_image = generate_image(img, model)
        generated_image = generated_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        generated_image = (generated_image * 255).astype(np.uint8)
        img_pil = Image.fromarray(generated_image)
        img_pil.save(os.path.join(config['generated_images_path'], f"generated_image_{i}.png"))

if __name__ == '__main__':
    save_generated_images()
