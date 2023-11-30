import utils
from torch.optim import Adam
from dataset import CustomDataset
import torch
from ddpm import Diffusion
from unet import UNet_Conditional
from torch.utils.data import DataLoader
from ddpm_utils import embedding_labels, get_loss
from torchvision import transforms


def main_train(model, diff_model, train_loader, batch_size=32, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = epochs

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            embed_labels = embedding_labels(labels)
            embed_labels = embed_labels.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, diff_model.noise_steps,
                              (batch_size,), device=device).long()
            loss = get_loss(model, diff_model, images, t, embed_labels)
            loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % 500 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item()} ")
                torch.save(model.state_dict(), "models/unet_2.pt")
                # sample_plot_image(model, diff_model, diff_vars, torch.t(embed_labels))


if __name__ == "__main__":
    IMAGE_DIRECTORY = "dataset/ExpwCleaned"
    LABEL_FILE = "dataset/label_expw.csv"

    LABEL_FILE_COLUMNS = ["Image name", "size", "neutral", "happiness", "surprise",
                          "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Adjust to your desired image size
        transforms.ToTensor(),
    ])

    label_data = utils.get_df(LABEL_FILE, LABEL_FILE_COLUMNS)
    dataset = CustomDataset(IMAGE_DIRECTORY, label_data, transform)

    batch_size = 32
    print("Dataset artifacts loaded")

    # The Dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True, num_workers=2, drop_last=True)

    # The Diffusion Model
    model = UNet_Conditional(time_dim=32)

    # Creating an instance of the diffusion model
    diff_model = Diffusion()
    # model.load_state_dict(torch.load("models/unet_2.pt"))
    main_train(model, diff_model, train_loader,
               batch_size=batch_size, epochs=200)
