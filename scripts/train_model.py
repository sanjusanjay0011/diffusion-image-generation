import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from models.diffusion_model import DiffusionModel
import os
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loader
def load_dataset(dataset_path, batch_size):
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
        ])
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
def train():
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    train_loader = load_dataset(config['dataset_path'], config['batch_size'])

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for data, _ in tqdm(train_loader):
            data = data.to(device)
            noisy_data = noise_image(data)
            optimizer.zero_grad()

            output = model(noisy_data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss/len(train_loader)}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config['model_checkpoint_path'], f"model_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    os.makedirs(config['model_checkpoint_path'], exist_ok=True)
    train()
