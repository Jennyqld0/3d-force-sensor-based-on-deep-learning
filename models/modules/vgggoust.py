import torch
import torch.nn as nn
import torchvision

class Ghost3x3Module(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
           # nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            #nn.BatchNorm2d(init_channels),
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=True),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
           # nn.Conv2d(init_channels, new_channels, dw_size,1, dw_size//2, groups=init_channels, bias=False),
            #nn.BatchNorm2d(new_channels),
            nn.Conv2d(init_channels, new_channels, dw_size,1, dw_size//2, groups=init_channels, bias=True),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.conv2_openration= nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 1,1,0, groups=init_channels, bias=True),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        x3 = self.conv2_openration(x1+x2)
        out = torch.cat([x1,x3], dim=1)
     #   out= out[:, :self.oup, :, :]
   #     out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]



class VGGNet(nn.Module):
    def __init__(self, block_nums,num_classes=1000):
        super(VGGNet, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Ghost3x3Module(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Ghost3x3Module(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out

def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums)
    return model

def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGGNet(block_nums)
    return model

if __name__ == '__main__':
    model = VGG19()
    print(model)

    input = torch.randn(1,3,224,224)
    out = model(input)
    print(out.shape)

