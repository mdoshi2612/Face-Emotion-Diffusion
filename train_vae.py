import utils
from dataset import CustomDataset
from torchvision import transforms
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None, device="cuda", **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.0001
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]


if __name__ == "__main__":
    IMAGE_DIRECTORY = "dataset/ExpwCleaned"
    LABEL_FILE = "dataset/label_expw.csv"

    LABEL_FILE_COLUMNS = ["Image name", "size", "neutral", "happiness", "surprise",
                          "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    label_data = utils.get_df(LABEL_FILE, LABEL_FILE_COLUMNS)
    dataset = CustomDataset(IMAGE_DIRECTORY, label_data, transform)

    print("Dataset artifacts loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = 3
    latent_dim = 2
    hidden_dims = [32, 64, 128, 256, 512]

    vae_model = VanillaVAE(in_channels, latent_dim,
                           hidden_dims, device).to(device)

    vae_model.train()
    print(vae_model)

    learning_rate = 3e-4
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

    batch_size = 32
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    num_epochs = 100
    log_interval = 500

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            results = vae_model(data)
            recons_batch = results[0]
            mu = results[2]
            logvar = results[3]

            loss_dict = vae_model.loss_function(recons_batch, data, mu, logvar)
            total_loss = loss_dict['loss']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item()}")

        print(f"Epoch {epoch + 1}, Loss: {total_loss.item()}")
        if (epoch + 1) % 5 == 0:
            torch.save(vae_model.state_dict(),
                       "models/weights_latent_dim_2.pt")

    print("Training finished.")
