import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


# class BasicBlock(nn.Module):
# 	def __init__(self, in_planes, planes, stride=1):
# 		super(BasicBlock, self).__init__()

# 		# 3x3 필터를 사용 (너비와 높이를 줄일 때는 stride 값 조절)
# 		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
# 		self.bn1 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

# 		# 3x3 필터를 사용 (패딩을 1만큼 주기 때문에 너비와 높이가 동일)
# 		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
# 		self.bn2 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

# 		self.shortcut = nn.Sequential() # identity인 경우
# 		if stride != 1: # stride가 1이 아니라면, Identity mapping이 아닌 경우
# 			self.shortcut = nn.Sequential(
# 				nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
# 				nn.BatchNorm2d(planes)
# 			)

# 	def forward(self, x):
# 		out = F.relu(self.bn1(self.conv1(x)))
# 		out = self.bn2(self.conv2(out))
# 		out += self.shortcut(x) # (핵심) skip connection
# 		out = F.relu(out)
# 		return out


# # ResNet 클래스 정의
# class ResNet(nn.Module):
# 	def __init__(self, block, num_blocks, num_classes=18):
# 		super(ResNet, self).__init__()
# 		self.in_planes = 64

# 		# 64개의 3x3 필터(filter)를 사용
# 		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 		self.bn1 = nn.BatchNorm2d(64)
# 		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
# 		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
# 		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
# 		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
# 		# self.linear = nn.Linear(512, num_classes)
# 		self.linear = nn.Linear(12288, num_classes)

# 	def _make_layer(self, block, planes, num_blocks, stride):
# 		strides = [stride] + [1] * (num_blocks - 1)
# 		layers = []
# 		for stride in strides:
# 			layers.append(block(self.in_planes, planes, stride))
# 			self.in_planes = planes # 다음 레이어를 위해 채널 수 변경
# 		return nn.Sequential(*layers)

# 	def forward(self, x):
# 		out = F.relu(self.bn1(self.conv1(x)))
# 		# print(out.shape)
# 		out = self.layer1(out)
# 		# print(out.shape)
# 		out = self.layer2(out)
# 		# print(out.shape)
# 		out = self.layer3(out)
# 		# print(out.shape)
# 		out = self.layer4(out)
# 		# print(out.shape)
# 		out = F.avg_pool2d(out, 4)
# 		# print(out.shape)
# 		out = out.view(out.size(0), -1)
# 		#print(out.shape)

# 		out = self.linear(out)
# 		# print(out.shape)
# 		return out



class ResNet(torchvision.models.ResNet):
	pass
