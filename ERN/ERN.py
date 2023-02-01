import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.filters import GaussianBlur2d, Laplacian, Sobel


class EFEBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_method: str = 'laplacian', down_sample: bool = True):
        super(EFEBlock, self).__init__()
        self.blur = GaussianBlur2d(kernel_size=(3, 3), sigma=(1., 1.), border_type='reflect')
        
        if edge_method in ['laplacian', 'l']:
            self.edge = Laplacian(kernel_size=3, border_type='reflect', normalized=False)
        elif edge_method in ['sobel', 's']:
            self.edge = Sobel(normalized=False)
        else:
            raise Exception("Invalid edge_method:", edge_method, "edge_method supports laplacian, sobel")
        
        if down_sample == True:
            self.conv3x3 = torch.nn.Conv2d
        else:
            self.conv3x3 = torch.nn.ConvTranspose2d
        
        
        self.blurConv = torch.nn.Sequential(
            self.conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.PReLU()
        )
        
        self.edgeConv = torch.nn.Sequential(
            self.conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.PReLU()
        )
        
        self.orignConv = torch.nn.Sequential(
            self.conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.PReLU()
        )
        
        self.conv = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        blur = self.blur(x)
        blur = self.blurConv(blur)
        
        edge = self.edge(x)
        edge = self.edgeConv(edge)
        
        origin = self.orignConv(x)
        
        out = blur + edge + origin
        out = self.conv(out)
    
        return out

    
class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num=1):
        super(SkipBlock, self).__init__()
        self.blocks = []
        self.num = num
        self.mid_channels = mid_channels
        self.blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1))
        
        for i in range(num - 1):
            self.blocks.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1))
        for i in range(num - 1):
            self.blocks.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1))
        
        self.blocks.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        
        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        short_cut = x
        x = self.layers(x)
        
        return x + short_cut

    
class Defed(nn.Module):
    def __init__(self, num_channels, edge_method: str = 'laplacian'):
        super(ERN, self).__init__()
        self.skip1 = SkipBlock(in_channels=num_channels, out_channels=num_channels, mid_channels=16, num=3)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.skip2 = SkipBlock(in_channels=16, out_channels=16, mid_channels=16, num=2)
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.PReLU()
        )
        
        self.en_efe_1 = EFEBlock(in_channels=16, out_channels=16, edge_method=edge_method, down_sample=True)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.skip3 = SkipBlock(in_channels=32, out_channels=32, mid_channels=32, num=1)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.de_efe_1 = EFEBlock(in_channels=16, out_channels=16, edge_method=edge_method, down_sample=False)
                
        self.en_efe_2 = EFEBlock(in_channels=32, out_channels=32, edge_method=edge_method, down_sample=True)
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.de_efe_2 = EFEBlock(in_channels=32, out_channels=32, edge_method=edge_method, down_sample=False)
    
    def forward(self, x):
        enskip1 = self.skip1(x)
        x = self.head(x)
        enskip2 = self.skip2(x)
        x = self.en_efe_1(x)
        x = self.down1(x)
        enskip3 = self.skip3(x)
        x = self.en_efe_2(x)
        x = self.down2(x)
        
        x = self.up2(x)
        x = self.de_efe_2(x) + enskip3
        x = self.up1(x)
        x = self.de_efe_1(x) + enskip2
        x = self.tail(x) + enskip1
        
        return x

    
if __name__ == '__main__':
    sample = torch.randn(1, 1, 28, 28).cuda()
    print("Mnist Test")
    print(ERN(num_channels=1, edge_method='laplacian').cuda()(sample).shape)
    print(ERN(num_channels=1, edge_method='sobel').cuda()(sample).shape)
    
    sample = torch.randn(1, 3, 32, 32).cuda()
    print("Cifar10 and SVHN Test")
    print(ERN(num_channels=3, edge_method='laplacian').cuda()(sample).shape)
    print(ERN(num_channels=3, edge_method='sobel').cuda()(sample).shape)
    
    sample = torch.randn(1, 3, 224, 224).cuda()
    print("Imagenette Test")
    print(ERN(num_channels=3, edge_method='laplacian').cuda()(sample).shape)
    print(ERN(num_channels=3, edge_method='sobel').cuda()(sample).shape)
