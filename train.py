import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from tensorboardX import SummaryWriter
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net, encoder_decoder

def main():
    parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer Model")
    parser.add_argument("-content_dir", type=str, default="datasets/COCO1K/", help="Path to content images directory")
    parser.add_argument("-style_dir", type=str, default="datasets/WikiArt1k/", help="Path to style images directory")
    parser.add_argument("-gamma", type=float, default=1.0, help="Alpha blending parameter for AdaIN")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-b", type=int, default=8, help="Batch size")
    parser.add_argument("-encoder_name", type=str, default="encoder.pth", help="Encoder model name")
    parser.add_argument("-decoder_name", type=str, default="decoder.pth", help="Decoder model name")
    parser.add_argument("-p", type=str, default="decoder.png", help="Path to save sample output image")
    parser.add_argument("-cuda", type=str, default="Y", help="Use CUDA (Y/N)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda == "Y" else 'cpu')

    sample_path = "samples/"
    save_dir = Path('./experiments')
    save_dir.mkdir(exist_ok=True, parents=True)
    sample_dir = Path(sample_path)
    sample_dir.mkdir(exist_ok=True, parents=True)

    model = AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)
    model = model.to(device)
    model.train()

    encoder_state_dict = torch.load(args.encoder_name, map_location=device)
    model.encoder.load_state_dict(encoder_state_dict)

    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    style_dataset = custom_dataset(dir=args.style_dir, transform=transform)
    style_loader = DataLoader(style_dataset, batch_size=args.b, shuffle=True, num_workers=4)  # Reduced num_workers

    content_dataset = custom_dataset(dir=args.content_dir, transform=transform)
    content_loader = DataLoader(content_dataset, batch_size=args.b, shuffle=True, num_workers=4)  # Reduced num_workers

    optimizer = optim.Adam(model.decoder.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    reconstruction_losses = []
    style_losses = []
    total_losses = []

    print("Start Training on ", device)
    for epoch in range(args.e):
        for i, (content_images, style_images) in enumerate(zip(content_loader, style_loader)):
            content_images, style_images = content_images.to(device), style_images.to(device)

            content_features = model.encode(content_images)
            style_features = model.encode(style_images)
            output = model.decode(content_features[-1])

            loss_reconstruction = torch.nn.MSELoss()(output, content_images)

            style_loss = 0.0
            for content_feat, style_feat in zip(content_features, style_features):
                style_loss += model.style_loss(content_feat, style_feat)
            num_elements = content_features[-1].size(1) * content_features[-1].size(2)
            style_loss /= num_elements

            style_loss_weight = 1e-4
            loss = loss_reconstruction + style_loss_weight * style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f'Epoch [{epoch + 1}/{args.e}], Step [{i + 1}/{len(content_loader)}], '
                  f'Reconstruction Loss: {loss_reconstruction.item():.4f}, Style Loss: {style_loss.item():.4f}')

            writer.add_scalar('loss_reconstruction', loss_reconstruction.item(), epoch * len(content_loader) + i + 1)
            writer.add_scalar('loss_style', style_loss.item(), epoch * len(content_loader) + i + 1)

            if (epoch * len(content_loader) + i + 1) % 10000 == 0 or (
                    epoch == args.e - 1 and i == len(content_loader) - 1):
                state_dict = model.decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir / f'decoder_iter_{epoch * len(content_loader) + i + 1}.pth')

            if (epoch * len(content_loader) + i + 1) % 1000 == 0:
                with torch.no_grad():
                    sample_output = model.decode(content_features[-1])
                    sample_output = sample_output.clamp(0, 1)
                    sample_filename = sample_dir / f'sample_epoch_{epoch}_iter_{i}.png'
                    torchvision.utils.save_image(sample_output, sample_filename, nrow=8)

        # Append the loss values to the lists
        reconstruction_losses.append(loss_reconstruction.item())
        style_losses.append(style_loss.item())
        total_losses.append(loss.item())

    writer.close()

    # Create a plot to visualize the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.plot(style_losses, label='Style Loss')
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()

    # Add a timestamp to the figure's name
    figure_name = f'loss_plot.png'

    # Save the plot to the "loss_plots" folder
    loss_plot_path = Path('./loss_plots')
    loss_plot_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(loss_plot_path / figure_name)

if __name__ == '__main__':
    main()
