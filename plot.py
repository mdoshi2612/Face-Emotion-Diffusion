import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def show_tensor_image(image, ax):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    ax.imshow(reverse_transforms(image))


def simulate_forward_diffusion(train_loader, diff_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = next(iter(train_loader))[0].to(device)

    num_images = 8
    stepsize = diff_model.noise_steps // num_images

    plt.figure(figsize=(15, 15))
    plt.axis('off')

    for idx in range(0, diff_model.noise_steps, stepsize):
        t = torch.Tensor([idx]).type(torch.int64).to(device)

        subplot_idx = idx // stepsize + 1
        plt.subplot(1, num_images+1, subplot_idx)

        img, noise = diff_model.forward_diffusion_sample(image, t)
        show_tensor_image(img.cpu())
