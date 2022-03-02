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


def set_outfeature(class_num,trained_model):
    model_ = trained_model(pretrained=True)

    model_.fc = torch.nn.Linear(in_features=model_.fc.in_features, out_features=class_num, bias=True)
    torch.nn.init.xavier_uniform_(model_.fc.weight)
    stdv = 1. / math.sqrt(model_.fc.weight.size(1))
    model_.fc.bias.data.uniform_(-stdv, stdv)

    return model_