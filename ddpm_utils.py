from models.ddpm import Diffusion
from models.unet import UNet_Conditional
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
from plot import show_tensor_image


def get_loss(model, diff_model, x_0, t, label):
    x_noisy, noise = diff_model.forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t, label)
    return F.l1_loss(noise, noise_pred)


def get_index_from_list(vals, t, x_shape, device="cuda"):
    t = t.to(device)
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionVariables:

    def __init__(self, diff_model):
        self.alphas_cumprod = torch.cumprod(diff_model.alpha, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / diff_model.alpha)

        self.posterior_variance = diff_model.beta * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


def sample_timestep(x, t, label, model, diff_model, diff_var):
    with torch.no_grad():
        betas_t = get_index_from_list(diff_model.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            diff_var.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(
            diff_var.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * \
            (x - betas_t * model(x, t, label) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(
            diff_var.posterior_variance, t, x.shape)

        if t == 0:
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


def sample_plot_image(model, diff_model, diff_var, label, IMG_SIZE=64, noise_steps=50, device="cuda"):
    with torch.no_grad():
        # Sample noise
        img_size = IMG_SIZE
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        num_images = 10
        stepsize = int(noise_steps/num_images)

        for i in range(0, noise_steps)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t, label, model, diff_model, diff_var)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                show_tensor_image(img.detach().cpu())
        plt.show()


def embedding_labels(labels):
    label_idx_list: list = []
    for label in labels:
        for idx, val in enumerate(label):
            if val != 0:
                label_idx_list.append(idx)

    return torch.tensor(label_idx_list)
