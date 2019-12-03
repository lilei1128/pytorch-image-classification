import numpy as np
import torch
import torchvision
from torch import nn,optim
from config import config



def l2_norm(x):
    norm = torch.norm(x,p =2 ,dim =1 ,keepdim= True)
    x = torch.div(x,norm)
    return x
# 自己搭建一个简单卷积神经网络
class myModel(nn.Module):
    def __init__(self,num_classes):
        super(myModel,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3), #in_channels out_channels kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2,stride = 2)  #149
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,3,2),  #74    #
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size =2,stride=2)  #37

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,3,2),  #18
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2, stride = 2)  #9
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2592,120),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(True),
            nn.Linear(84,num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class ResNet18(nn.Module):
    def __init__(self,model,num_classes = 1000):
        super(ResNet18,self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(512,1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,num_classes)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x= self.backbone.bn1 (x)
        x = self.backbone.relu(x)
        x= self.backbone.maxpool(x)

        x= self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x= self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x= x.view(x.size(0),-1)
        x= l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class ResNet101(nn.Module):
    def __init__(self,model,num_classes =1000):
        super(ResNet101,self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048,2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048,num_classes)

    def forward(self,x):
        x = self.backbone.conv1(x)
        x =  self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0),-1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def get_net():
    #backbone = torchvision.models.resnet18(pretrained=True)
    #models = ResNet18(backbone,config.num_classes)
    backbone = torchvision.models.resnet101(pretrained=True)
    models = ResNet101(backbone, config.num_classes)
    return models