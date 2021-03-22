
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#데이터 로드 및 선처리
 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(110),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        transforms.Normalize([0.485, 0.485, 0.406], [0.229, 0.224,0.225])
    ])
    'val' : transforms.Compose([
        transforms.Resize(128),
        transforms.Centercrop(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.29, 0.224, 0.225])
    ])
}

#train과 val에 대한 데이터를 딕셔너리 형태로 저장한다.
data_dir = 'C:/workspace/lotte/'
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                    data_transforms[x])
        for x in ['train','val']}
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
               shuffle=True, num_workers=4)}
dataset_sizes = {x:len(image_datasets[x])for x in['train','val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuad.is_available() else"CPU")

#이미지 시각화! => numpy 배열로 바꾸자
def imshow(inp, title=None):
    