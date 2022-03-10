import torch
import torchvision.transforms as transforms
from torchvision.transforms import *
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import albumentations.pytorch
from albumentations import Compose, OneOf, Resize, Normalize

####################################transforms##########################################
def train_transform(resize=(512, 384)):
    transform = Compose([
        Resize(resize[0], resize[1]),
        A.HorizontalFlip(),
        OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
            ], p = 0.2),
        OneOf([
            A.MotionBlur(blur_limit = 3, p = 0.2),
            A.MedianBlur(blur_limit = 3, p = 0.1),
            A.Blur(blur_limit = 3, p = 0.1),
            ], p = 0.2),
        A.ShiftScaleRotate(rotate_limit = 15),
        OneOf([
            A.OpticalDistortion(p = 0.3),
            A.GridDistortion(p = 0.1),
            A.IAAPiecewiseAffine(p = 0.3),
            ], p = 0.2),
        OneOf([
            A.CLAHE(clip_limit = 2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
            ], p = 0.3),
        A.HueSaturationValue(p = 0.3),
        Normalize(),
        albumentations.pytorch.ToTensor(),
    ])
    return transform

def val_transform(resize=(512, 384)):
    transform = Compose([
            Resize(resize[0], resize[1]),
            Normalize(),
            albumentations.pytorch.ToTensor(),
        ])
    return transform


####################################### data sets #####################################################
class train_dataset(Dataset):
    def __init__(self, df, target, transform = None, one_hot=False):
        super(train_dataset, self).__init__()
        self.x = df['path']
        self.y = df[target]
        self.transform = transform
        self.one_hot = one_hot
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = np.array(Image.open(self.x.iloc[idx]))
        if self.one_hot:
            y = np.eye(10)[self.y.iloc[idx]]         # onehot ending (단위행렬의 n번째 열)
        else:
            y = self.y.iloc[idx]

        if self.transform:
            img = self.transform(image = X)['image']
        else:
            img = X
        return img, torch.tensor(y)
    

class test_dataset(Dataset):
    def __init__(self, imgs, transform = None, n_tta = None):
        super(test_dataset, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.n_tta = n_tta
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        X = np.array(Image.open(self.imgs[idx]))
        if self.transform:
            if self.n_tta:
                imgs = [self.transform(image = X)['image'] for _ in range(self.n_tta)]
                return imgs
            else:
                img = self.transform(image = X)['image']
                return img
        else:
            return X