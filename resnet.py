import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.id = nn.Sequential()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(),
                                  nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(planes))

        if stride != 1 or in_planes != planes:
            self.id = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.id(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        layers1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        layers2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        layers3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        layers4 = self._make_layers(block, 512, num_blocks[3], stride=2)
        
        self.layers = layers1 + layers2 + layers3 + layers4
        
        self.layer1 = nn.Sequential(*layers1)
        self.layer2 = nn.Sequential(*layers2)
        self.layer3 = nn.Sequential(*layers3)
        self.layer4 = nn.Sequential(*layers4)
        self.linear = nn.Linear(512, num_classes)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return layers
    
    def feature_layer(self, k, x):
        return(self.layers[k](x))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)