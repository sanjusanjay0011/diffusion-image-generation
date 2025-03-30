import torch
import torch.nn.functional as F

def noise_image(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    return image + noise

def denoise(image, model):
    with torch.no_grad():
        return model(image)
