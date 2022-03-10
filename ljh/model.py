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


# pretrained model 변형
def set_outfeature(class_num,trained_model):
    model_ = trained_model(pretrained=True)

    model_.fc = torch.nn.Linear(in_features=model_.fc.in_features, out_features=class_num, bias=True)
    torch.nn.init.xavier_uniform_(model_.fc.weight)
    stdv = 1. / math.sqrt(model_.fc.weight.size(1))
    model_.fc.bias.data.uniform_(-stdv, stdv)

    return model_


# resnet34
class Conv_Block_x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_Block_x2, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride, padding = (1, 1), bias = False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), 1, padding = (1, 1), bias = False)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), stride, bias = False)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        
        x += shortcut
        outputs = self.relu(x)
        return outputs
    
class Identity_Block_x2(nn.Module):
    def __init__(self, channels):
        super(Identity_Block_x2, self).__init__()

        self.conv_1 = nn.Conv2d(channels, channels, (1, 1), bias = False)
        self.bn_1 = nn.BatchNorm2d(channels)
        
        self.conv_2 = nn.Conv2d(channels, channels, (3, 3), 1, padding = (1, 1), bias = False)
        self.bn_2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        x += inputs
        outputs = self.relu(x)
        return outputs
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), 2, padding = (3, 3), bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, padding = (1, 1)),
        )
        
        self.block_2 = nn.Sequential(
            Identity_Block_x2(64),
            Identity_Block_x2(64),
            Identity_Block_x2(64),
        )
        
        self.block_3 = nn.Sequential(
            Conv_Block_x2(64, 128, 2),
            Identity_Block_x2(128),
            Identity_Block_x2(128),
            Identity_Block_x2(128),
        )
        
        self.block_4 = nn.Sequential(
            Conv_Block_x2(128, 256, 2),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
        )
        
        self.block_5 = nn.Sequential(
            Conv_Block_x2(256, 512, 2),
            Identity_Block_x2(512),
            Identity_Block_x2(512),
        )
        
        self.classifier = nn.Linear(512, 10)
        
    def forward(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = torch.mean(x, axis = [2, 3])
        outputs = self.classifier(x)
        return outputs