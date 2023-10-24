import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from custom_dataset import custom_dataset
import AdaIN_net as net
from torch.optim.lr_scheduler import StepLR


def main():
    parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer Model")
    parser.add_argument("-content_dir", type=str, default="datasets/COCO1K/", help="Path to content images directory")
    parser.add_argument("-style_dir", type=str, default="datasets/WikiArt1k/", help="Path to style images directory")
    parser.add_argument("-gamma", type=float, default=1.0, help="Alpha blending parameter for AdaIN")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-b", type=int, default=20, help="Batch size")
    parser.add_argument("-l", type=str, default="encoder.pth", help="Encoder model name")
    parser.add_argument("-s", type=str, default="decoder.pth", help="Decoder model name")
    parser.add_argument("-p", type=str, default="decoder.png", help="Path to save sample output image")
    parser.add_argument("-cuda", type=str, default="Y", help="Use CUDA (Y/N)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda == "Y" else 'cpu')

    save_dir = Path('./experiments')
    save_dir.mkdir(exist_ok=True, parents=True)

    # initialize model
    decoder = net.encoder_decoder.decoder
    encoder = net.encoder_decoder.encoder

    # load encoder model
    encoder.load_state_dict(torch.load(args.l))

    model = net.AdaIN_net(encoder, decoder)
    model = model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])

    style_dataset = custom_dataset(dir=args.style_dir, transform=transform)
    style_loader = DataLoader(style_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    content_dataset = custom_dataset(dir=args.content_dir, transform=transform)
    content_loader = DataLoader(content_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3, weight_decay=0.2)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    content_losses = []
    style_losses = []
    total_losses = []

    print("Now training on ", device)
    for epoch in range(args.e):
        for i, (content_data, style_data) in enumerate(zip(content_loader, style_loader)):
            content_images = content_data.to(device)
            style_images = style_data.to(device)

            # forward pass
            content_loss, style_loss = model(content_images, style_images, args.gamma)

            # calculate total loss
            total_loss = content_loss + style_loss

            # zero the gradients
            optimizer.zero_grad()

            # backpropagation
            total_loss.backward()
            optimizer.step()

            # print loss statistics
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.e}] Iteration [{i}/{len(content_loader)}]")
                print(f"Content Loss: {content_loss.item():.4f} Style Loss: {style_loss.item():.4f}")

            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            total_losses.append(total_loss.item())

        # update the learning rate
        scheduler.step()

    # save the trained model
    torch.save(model.decoder.state_dict(), args.s)

    # create a plot to visualize loss values
    plt.figure(figsize=(10, 6))
    plt.plot(content_losses, label='Content Loss')
    plt.plot(style_losses, label='Style Loss')
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()

    figure_name = f'{args.p}'

    # Save the plot to the "loss_plots" folder
    loss_plot_path = Path('./loss_plots')
    loss_plot_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(loss_plot_path / figure_name)


if __name__ == '__main__':
    main()
