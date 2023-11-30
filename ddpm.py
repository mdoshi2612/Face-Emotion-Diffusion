import torch

class Diffusion():
  def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=48, num_classes=7, c_in=3, c_out=3, device="cuda"):
    #Params
    self.img_size = img_size
    self.device = device
    self.c_in = c_in
    self.num_classes = num_classes

    #Define Beta
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.beta = self.linear_beta_schedule(self.noise_steps, self.beta_start, self.beta_end).to(device)

    #Define Alpha
    self.alpha = 1. - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0) #cumulative product

  def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
      return torch.linspace(start, end, timesteps)

  def sample_timesteps(self, n):
      return torch.randint(low=1, high=self.noise_steps, size=(n,))

  def forward_diffusion_sample(self, x, t, device = "cuda"):
      """
      Takes an image and a timestep as input and
      returns the noisy version of it
      """
      sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(device)
      sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(device)
      Ɛ = torch.randn_like(x)
      return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ