import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision import transforms
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net, encoder_decoder

# Define command-line arguments
parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer Model")
parser.add_argument("-content_dir", type=str, default="datasets/COCO1K/", help="Path to content images directory")
parser.add_argument("-style_dir", type=str, default="datasets/WikiArt1k/", help="Path to style images directory")
parser.add_argument("-gamma", type=float, default=1.0, help="Alpha blending parameter for AdaIN")
parser.add_argument("-e", type=int, default=20, help="Number of epochs")
parser.add_argument("-b", type=int, default=20, help="Batch size")
parser.add_argument("-encoder_name", type=str, default="encoder.pth", help="Encoder model name")
parser.add_argument("-decoder_name", type=str, default="decoder.pth", help="Decoder model name")
parser.add_argument("-p", type=str, default="decoder.png", help="Path to save sample output image")
parser.add_argument("-cuda", type=str, default="Y", help="Use CUDA (Y/N)")
args = parser.parse_args()

def main():
    # Set device to 'cuda' if available, otherwise 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda == "Y" else 'cpu')

    # Create a directory to save model checkpoints and logs
    save_dir = Path('./experiments')
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Load the encoder from the provided file
    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.encoder_name, map_location=device))

    # Load the decoder from the provided file (if available)
    if Path(args.decoder_name).exists():
        decoder = encoder_decoder.decoder
        decoder.load_state_dict(torch.load(args.decoder_name, map_location=device))
    else:
        # Create a new decoder with random weights if not provided
        decoder = encoder_decoder.decoder

    decoder = decoder.to(device)
    decoder.train()

    # Define the content transformation (you can customize this)
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    # Create a custom dataset for content images
    content_dataset = custom_dataset(dir=args.content_dir, transform=transform)
    content_loader = data.DataLoader(content_dataset, batch_size=args.b, shuffle=True, num_workers=16)

    # Define the optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(args.e):
        for i, content_images in enumerate(content_loader):
            content_images = content_images.to(device)

            # Forward pass through the decoder
            output = decoder(content_images)

            # Calculate the reconstruction loss (MSE loss)
            loss = torch.nn.MSELoss()(output, content_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss_reconstruction', loss.item(), epoch * len(content_loader) + i + 1)

            # Save the decoder's weights periodically
            if (epoch * len(content_loader) + i + 1) % 10000 == 0 or (epoch == args.e - 1 and i == len(content_loader) - 1):
                state_dict = decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir / f'decoder_iter_{epoch * len(content_loader) + i + 1}.pth.tar')

    writer.close()

if __name__ == '__main__':
    main()
