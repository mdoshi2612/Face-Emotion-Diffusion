import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_folder, dataframe, transform=None):
        self.data = dataframe
        self.data_folder = data_folder
        self.transform = transform

        self.image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_files[idx])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 2:9].values.astype("float")

        if self.transform:
            image = self.transform(image)

        return image, label
