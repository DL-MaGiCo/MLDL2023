import mixvpr
import torch
import torch.nn as nn
import torchvision

class ResNetMixVPR(nn.Module):
  def __init__(self):
        super(ResNetMixVPR, self).__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None
        self.aggregator = mixvpr.MixVPR()
    
  def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.aggregator(x)
        return x