import argparse
import torch
import torchvision.transforms as transforms
import AdaIN_net as net
from custom_dataset import custom_dataset
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt


# class DataStruct:
#     """
#     Structure used to swap between train and validation datasets in the dataloader
#     This is used to decrease the amount of memory used by the gpu, this avoids having to create 4
#     """
#     def __init__(self, train_dir: str, val_dir: str, batch_size: int, transform, shuffle: bool):
#         self.train_dataset = custom_dataset(train_dir, transform())
#         self.val_dataset = custom_dataset(val_dir, transform())
#         self.dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

#     def set_to_train(self):
#         self.dataloader.dataset = self.train_dataset

#     def set_to_val(self):
#         self.dataloader.dataset = self.val_dataset


def save_loss_plot(losses_train: list, losses_c: list, losses_s: list, losses_val: list, save_path: str):
    # Plot training losses
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')
    plt.plot([i for i in range(len(losses_c))], losses_c, label='Content Loss')
    plt.plot([i for i in range(len(losses_s))], losses_s, label='Style Loss')
    # Plot validation losses
    plt.plot([i for i in range(len(losses_val))], losses_val, label='Validation Loss', linestyle='--')

    # Set the title and labels
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def train_transform(img_size=512):
    return transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def validate(val_content_loader: DataLoader, val_style_loader: DataLoader, model: net.AdaIN_net, device):
    """
    Originally used to validate the model but is unused due to it being very memory hungry and refusing to run on my laptop
    """
    assert len(val_content_loader) == len(val_style_loader)
    loss_validate = 0.0
    for c_img, s_img in zip(val_content_loader, val_style_loader):
        c_img = c_img.to(device)
        s_img = s_img.to(device)
        loss_c, loss_s = model(c_img, s_img)
        loss_validate += loss_c.item() + loss_s.item()

        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
    avg_loss = loss_validate / len(val_content_loader)
    return avg_loss



def train(n_epochs: int,
          optimizer: Optimizer,
          model: net.AdaIN_net,
          content_loader: DataLoader,
          style_loader: DataLoader,
          val_content_loader: DataLoader,
          val_style_loader: DataLoader,
          scheduler: LRScheduler,
          device: torch.device,
          alpha: float):
    assert len(content_loader) == len(style_loader)

    current_time = datetime.datetime.now()

    # Format the current time as a string
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print("Training")
    model.train()
    losses_train = []
    losses_content = []
    losses_style = []

    losses_val = []
    for epoch in range(1, n_epochs + 1):
        print(f"[{timestamp}] Epoch: {epoch}")
        loss_train = 0.0
        loss_content = 0.0
        loss_style = 0.0
        for c_imgs, s_imgs in zip(content_loader, style_loader):
            c_imgs = c_imgs.to(device)
            s_imgs = s_imgs.to(device)

            loss_c, loss_s = model(c_imgs, s_imgs, alpha)
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            loss_content += loss_c.item()
            loss_style += loss_s.item()
        scheduler.step()

        size = len(content_loader)
        losses_train += [loss_train / size]
        losses_content += [loss_content / size]
        losses_style += [loss_style / size]

        validation_loss = validate(val_content_loader, val_style_loader, model, device)

        losses_val.append(validation_loss)
        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}, Training Loss: {loss_train / size}, Content Loss: {loss_content / size}, Style Loss: {loss_style / size}, Validation Loss: {validation_loss}"
        print(training_str)
    return losses_train, losses_content, loss_style, losses_val


def main(args):
    # Initialize device
    device = get_device(args.cuda)
    print("Device: ", device)
    # Load data
    content_dataset = custom_dataset(args.content_dir, train_transform())
    style_dataset = custom_dataset(args.style_dir, train_transform())

    content_loader = DataLoader(content_dataset, batch_size=args.b, shuffle=True)
    style_loader = DataLoader(style_dataset, batch_size=args.b, shuffle=True)

    validation_content_dataset = custom_dataset(args.content_val_dir, train_transform())
    validation_style_dataset = custom_dataset(args.style_val_dir, train_transform())

    val_content_loader = DataLoader(validation_content_dataset, batch_size=1, shuffle=False)
    val_style_loader = DataLoader(validation_style_dataset, batch_size=1, shuffle=False)

    # Initialize model
    decoder = net.encoder_decoder.decoder
    encoder = net.encoder_decoder.encoder

    # Load ImageNet model
    encoder.load_state_dict(torch.load(args.l))
    # encoder = nn.Sequential(*list(encoder.children())[:31])

    model = net.AdaIN_net(encoder, decoder)
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.2)
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = 0.0001
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    # Setup Train loop
    losses_train, losses_c, losses_s, losses_val = train(args.e, optimizer, model, content_loader, style_loader,
                                                         val_content_loader, val_style_loader, scheduler, device,
                                                         args.gamma)
    save_loss_plot(losses_train, losses_c, losses_s, losses_val, args.p)
    # torch.save(model.state_dict(), "model.pth")

    torch.save(decoder.state_dict(), args.s)
    print("Training is complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-content_dir', type=str, help='Directory containing content images')
    parser.add_argument('-style_dir', type=str, help='Directory containing style content')
    parser.add_argument('-content_val_dir', type=str, default="./datasets/COCO100/",
                        help='Directory containing validation images for the content')
    parser.add_argument('-style_val_dir', type=str, default='./datasets/wikiart100/',
                        help='Directory containing validation images for the style')
    parser.add_argument('-gamma', type=float, default=1.0, help='Heat Ratio')
    parser.add_argument('-e', type=int, help='Number of epochs')
    parser.add_argument('-b', type=int, help='Batch Size')
    parser.add_argument('-l', type=str, help='Encoder path')
    parser.add_argument('-s', type=str, help='Decoder path')
    parser.add_argument('-p', type=str, help='Loss Plot Path')
    parser.add_argument('-cuda', type=str, default='Y', help="Whether to use CPU or Cuda, use Y or N")
    args = parser.parse_args()
    main(args)