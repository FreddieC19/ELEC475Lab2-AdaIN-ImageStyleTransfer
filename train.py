import argparse
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net


# Define command-line arguments
parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer Model")
parser.add_argument("-content_dir", type=str, required=True, help="Path to content images directory")
parser.add_argument("-style_dir", type=str, required=True, help="Path to style images directory")
parser.add_argument("-gamma", type=float, default=1.0, help="Alpha blending parameter for AdaIN")
parser.add_argument("-e", type=int, default=20, help="Number of epochs")
parser.add_argument("-b", type=int, default=20, help="Batch size")
parser.add_argument("-l", type=str, default="encoder.pth", help="Encoder model name")
parser.add_argument("-s", type=str, default="decoder.pth", help="Decoder model name")
parser.add_argument("-p", type=str, default="decoder.png", help="Path to save sample output image")
parser.add_argument("-cuda", type=str, default="Y", help="Use CUDA (Y/N)")

args = parser.parse_args()

# Check if CUDA should be used
device = torch.device("cuda" if args.cuda.upper() == "Y" and torch.cuda.is_available() else "cpu")

# Define transformations for content and style images
content_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

style_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Create custom datasets for content and style images
content_dataset = custom_dataset(dir=args.content_dir, transform=content_transform)
style_dataset = custom_dataset(dir=args.style_dir, transform=style_transform)

# Create data loaders for content and style datasets
content_loader = DataLoader(content_dataset, batch_size=args.b, shuffle=True, num_workers=4)
style_loader = DataLoader(style_dataset, batch_size=args.b, shuffle=True, num_workers=4)

# Initialize the AdaIN_net model
encoder = args.l
model = AdaIN_net(encoder=encoder)

# Move the model to the appropriate device
model.to(device)

# Define the optimizer (you can use different optimizers and learning rates)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(args.e):
    model.train()

    # Iterate through both data loaders alternately
    for batch_idx, (content, _), (style, _) in zip(content_loader, style_loader):
        content, style = content.to(device), style.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        loss_c, loss_s = model(content, style, alpha=args.gamma)
        total_loss = loss_c + loss_s

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{args.e}] Batch [{batch_idx + 1}/{len(content_loader)}] Loss: {total_loss.item()}")

# Save decoder
torch.save(model.decoder.state_dict(), args.s)

# Generate a sample output image
sample_content, sample_style = next(iter(content_loader)), next(iter(style_loader))
sample_content = sample_content[0].unsqueeze(0).to(device)  # Take the first sample
sample_style = sample_style[0].unsqueeze(0).to(device)  # Take the first sample
output = model(sample_content, sample_style, alpha=args.gamma)
output = output.squeeze(0).cpu().detach().numpy()  # Convert to numpy array
output = output.transpose(1, 2, 0)  # Change channel order if needed

# Save the sample output image
plt.imsave(args.p, output)

print("Training completed and model checkpoints saved!")
