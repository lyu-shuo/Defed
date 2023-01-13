import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, method: str = 'Conv'):
        super(conv3x3, self).__init__()
        
        if method not in ['Conv', 'transConv']:
            raise Exception("Invalid Convolution Method : ", method, "Convolution Method supports Conv, transConv")
        elif method == 'Conv':
            self.conv3x3 = torch.nn.Conv2d
        elif method == 'transConv':
            self.conv3x3 = torch.nn.ConvTranspose2d
        
        self.conv = torch.nn.Sequential(
            self.conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

    
class BaseNet(nn.Module):
    def __init__(self, num_channels):
        super(BaseNet, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        
        self.body = nn.Sequential(
            conv3x3(in_channels=16, out_channels=16, method = 'Conv'),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            conv3x3(in_channels=32, out_channels=32, method = 'Conv'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            conv3x3(in_channels=32, out_channels=32, method = 'transConv'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            conv3x3(in_channels=16, out_channels=16, method = 'transConv'),
        )
        
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x