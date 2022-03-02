import torch
import torchvision.transforms as transforms
from torchvision.transforms import *
from PIL import Image
from torch.utils.data import Dataset
import albumentations


####################################transforms##########################################
class BaseAugmentation:
    def __init__(self,resize=(512, 384), mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class albumentations_transform:
    def __init__(self,resize=(512, 384), mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)):
        self.transform = albumentations.Compose([
            albumentations.Resize(resize[0],resize[1]), 
            albumentations.OneOf([
                          albumentations.MotionBlur(p=1),
                          albumentations.OpticalDistortion(p=1),
                          albumentations.GaussNoise(p=1)                 
            ], p=0.3),
            albumentations.OneOf([
                          albumentations.MotionBlur(p=1),
                          albumentations.OpticalDistortion(p=1),
                          albumentations.GaussNoise(p=1)                 
            ], p=0.3),
            albumentations.OneOf([
                          albumentations.MotionBlur(p=1),
                          albumentations.OpticalDistortion(p=1),
                          albumentations.GaussNoise(p=1)                 
            ], p=0.3),
            albumentations.Normalize(mean=mean, std=std),
            albumentations.pytorch.transforms.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)


####################################### data sets #####################################################
class CustomDataset(Dataset):
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

    def read_image(self, idx):
        return Image.open(self.x.iloc[idx])