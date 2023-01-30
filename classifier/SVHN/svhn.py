import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

'''
Backbone: VGGNet-11
Trained Parameter: 
Input Size: [B, 3, 32, 32]
Best Accuracy: 93.26%
Best Loss: 0.5212776340617269
'''

class SVHNClassifier(nn.Module):
    def __init__(self):
        super(SVHNClassifier, self).__init__()
        self.structure = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        self.features = self._make_layers(self.structure)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# class SVHNClassifier(nn.Module):
#     def __init__(self):
#         super(SVHNClassifier, self).__init__()
#         self.backbone = torchvision.models.vgg11()
#         self.fc = nn.Linear(in_features=1000, out_features=10, bias=True)
    
#     def forward(self, x):
#         x = self.backbone(x)
#         x = F.relu(x)
#         x = self.fc(x)
    
#         return x