import dataset

import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from sklearn.model_selection import train_test_split

random_seed = 11

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True # 고정하면 학습이 느려진다고 합니다.
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# params
test_name = 'T1'
targets = ['mask','sex','age']
batch_size = 128
MASK_CLASS_NUMs = [3,2,3]
LEARNING_RATE = 0.0001 # 학습 때 사용하는 optimizer의 학습률 옵션 설정
NUM_EPOCH = 10 # 학습 때 mnist train 데이터 셋을 얼마나 많이 학습할지 결정하는 옵션
loss_fn = torch.nn.CrossEntropyLoss()


def createDirectory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
            print("Error: Failed to create the directory.")


for model_idx in range(3):
    # params
    target = targets[model_idx]
    MASK_CLASS_NUM = MASK_CLASS_NUMs[model_idx]

    createDirectory(f'/opt/ml/level1_cv11/ljh/model/{test_name}_{target}_state_dict')

    id = pd.read_csv('/opt/ml/level1_cv11//input/data/train/train.csv')
    train_id, valid_id = train_test_split(id['id'], test_size=0.2, random_state=random_seed)
    df = pd.read_csv('/opt/ml/level1_cv11//input/data/train/train_new4.csv')

    transform = dataset.BaseAugmentation()
    train_data = dataset.MaskDataset(df[df['id'].isin(train_id)],target,transform)
    valid_data = dataset.MaskDataset(df[df['id'].isin(valid_id)],target,transform)
    train_iter = DataLoader(train_data,batch_size=batch_size//2,shuffle=True)
    valid_iter = DataLoader(valid_data,batch_size=batch_size,shuffle=True)

    transform_aug = dataset.CustomAugmentation()
    train_data_aug = dataset.MaskDataset(df[df['id'].isin(train_id)],target,transform_aug)
    train_iter_aug = DataLoader(train_data_aug,batch_size=batch_size//2,shuffle=True)

    import math
    # ImageNet에서 학습된 ResNet 18 딥러닝 모델을 불러옴
    imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)

    # 분류 모델의 output 크기가 1000개로 되어 있음으로 mnist data class 개수로 나올 수 있도록 Fully Connected Layer를 변경하고 xavier uniform으로 weight 초기화
    imagenet_resnet18.fc = torch.nn.Linear(in_features=512, out_features=MASK_CLASS_NUM, bias=True)
    torch.nn.init.xavier_uniform_(imagenet_resnet18.fc.weight)
    stdv = 1. / math.sqrt(imagenet_resnet18.fc.weight.size(1))
    imagenet_resnet18.fc.bias.data.uniform_(-stdv, stdv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

    imagenet_resnet18.to(device) # Resnent 18 네트워크의 Tensor들을 GPU에 올릴지 Memory에 올릴지 결정함
    optimizer = torch.optim.Adam(imagenet_resnet18.parameters(), lr=LEARNING_RATE) # weight 업데이트를 위한 optimizer를 Adam으로 사용함

    from tqdm import tqdm

    ### 학습 코드 시작
    best_test_accuracy = 0.
    best_test_loss = 9999.

    for epoch in range(1,NUM_EPOCH+1):
        for phase in ["train", "test"]:
            running_loss = 0.
            running_acc = 0.

            ###############################################################################################################
            if phase == "train":
                imagenet_resnet18.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
                for ind, (a,b) in enumerate(tqdm(zip(train_iter,train_iter_aug))):
                        images = torch.cat((a[0],b[0])).to(device)
                        labels = torch.cat((a[1],b[1])).to(device)

                        optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                        with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                            logits = imagenet_resnet18(images)
                            _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
                            loss = loss_fn(logits, labels)

                            loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                            optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

                        running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                        running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
                
                epoch_loss = running_loss / (len(train_iter.dataset) + len(train_iter_aug.dataset))
                epoch_acc = running_acc / (len(train_iter.dataset) + len(train_iter_aug.dataset))
            
            elif phase == "test":
                imagenet_resnet18.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함
                for ind,a in enumerate(tqdm(valid_iter)):
                        images = a[0].to(device)
                        labels = a[1].to(device)

                        optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                        with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                            logits = imagenet_resnet18(images)
                            _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
                            loss = loss_fn(logits, labels)

                        running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                        running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

            epoch_loss = running_loss / len(valid_iter.dataset)
            epoch_acc = running_acc / len(valid_iter.dataset)
            ##########################################################################################################################

            # 한 epoch이 모두 종료되었을 때,
            print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
            if phase == "test" and best_test_accuracy < epoch_acc: # phase가 test일 때, best accuracy 계산
                best_test_accuracy = epoch_acc
                torch.save(imagenet_resnet18.state_dict(), f'/opt/ml/level1_cv11/ljh/model/{test_name}_{target}_state_dict/{epoch:03d}_{epoch_acc*100%100:03.2f}.pt')
            if phase == "test" and best_test_loss > epoch_loss: # phase가 test일 때, best loss 계산
                best_test_loss = epoch_loss

    print("학습 종료!")
    print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}")
    torch.save(imagenet_resnet18.state_dict(), f'/opt/ml/level1_cv11/ljh/model/{target}.pt')
    torch.cuda.empty_cache() # GPU 캐시 데이터 삭제