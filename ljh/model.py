import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BaseModel(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d((2,2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, class_num)

    def forward(self, x):       # 32
        x = self.conv1(x)       # 30
        x = F.relu(x)

        x = self.conv2(x)       # 28
        x = self.pool(x)        # 14
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv3(x)       # 12
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


def load_resnet18(class_num):
    # ImageNet에서 학습된 ResNet 18 딥러닝 모델을 불러옴
    imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)

    # 분류 모델의 output 크기가 1000개로 되어 있음으로 mnist data class 개수로 나올 수 있도록 Fully Connected Layer를 변경하고 xavier uniform으로 weight 초기화
    imagenet_resnet18.fc = torch.nn.Linear(in_features=512, out_features=class_num, bias=True)
    torch.nn.init.xavier_uniform_(imagenet_resnet18.fc.weight)
    stdv = 1. / math.sqrt(imagenet_resnet18.fc.weight.size(1))
    imagenet_resnet18.fc.bias.data.uniform_(-stdv, stdv)

    print("네트워크 필요 입력 채널 개수", imagenet_resnet18.conv1.weight.shape[1])
    print("네트워크 출력 채널 개수 (예측 class type 개수)", imagenet_resnet18.fc.weight.shape[0])
    return imagenet_resnet18

def load_resnet34(class_num):
    # ImageNet에서 학습된 ResNet 18 딥러닝 모델을 불러옴
    imagenet_resnet18 = torchvision.models.resnet34(pretrained=True)

    # 분류 모델의 output 크기가 1000개로 되어 있음으로 mnist data class 개수로 나올 수 있도록 Fully Connected Layer를 변경하고 xavier uniform으로 weight 초기화
    imagenet_resnet18.fc = torch.nn.Linear(in_features=512, out_features=class_num, bias=True)
    torch.nn.init.xavier_uniform_(imagenet_resnet18.fc.weight)
    stdv = 1. / math.sqrt(imagenet_resnet18.fc.weight.size(1))
    imagenet_resnet18.fc.bias.data.uniform_(-stdv, stdv)

    print("네트워크 필요 입력 채널 개수", imagenet_resnet18.conv1.weight.shape[1])
    print("네트워크 출력 채널 개수 (예측 class type 개수)", imagenet_resnet18.fc.weight.shape[0])
    return imagenet_resnet18