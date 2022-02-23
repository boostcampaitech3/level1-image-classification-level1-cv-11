import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader 

class MaskDataset(Dataset):
    def __init__(self,df,target,transform = None):
        self.X = df['path']
        self.y = df[target]
        self.transform = transform
        return

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.X.iloc[idx])
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.y.iloc[idx])