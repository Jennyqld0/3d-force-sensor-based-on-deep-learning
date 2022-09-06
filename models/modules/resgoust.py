
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
__all__ = ['Res2Net', 'res2net50']


model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
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
        
class Ghost2Module(nn.Module):
    expansion = 1
    def __init__(self, inp,oup,stride=1, relu=True,downsample=False):

        super(Ghost2Module, self).__init__()
        self.midp = oup
        self.stride=stride
        self.goust1=GhostModule(inp, self.midp, kernel_size=3,stride=self.stride)
        self.goust2=GhostModule(self.midp, oup, kernel_size=3)

        # shortcut
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.goust1(x)
        out = self.goust2(out)
        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inp,oup,  stride=1, relu=True,downsample=False):

        super(Bottleneck, self).__init__()
        self.midp = oup
        self.stride=stride

        self.goust1=GhostModule(inp, self.midp, kernel_size=1)
        self.goust2=GhostModule(self.midp, oup, kernel_size=3,stride=self.stride)
        self.goust3=GhostModule(self.midp, oup * self.expansion, kernel_size=1)
        

        # shortcut
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup* self.expansion, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(oup* self.expansion),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        residual = x
        out = self.goust1(x)
        out = self.goust2(out)
        out = self.goust3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
        
class Ghost2Net(nn.Module):

    def __init__(self, block, layers,  num_classes=3):
        self.inplanes = 64
        super(Ghost2Net, self).__init__()
        self.gous1=GhostModule(3, 64, kernel_size=7,stride=2)
     #   self.gous2=GhostModule(64, 64, kernel_size=3,stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion, 100)
        self.fc1 = nn.Linear(100, num_classes)
      #  self.dropout = nn.Dropout(p=0.2)


    def _make_layer(self, block,planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes,  planes, stride=stride,downsample=True))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,  planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gous1(x)
        x = self.maxpool(x)
     #   x = self.gous2(x)
    #    x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
   #     print(x.shape)

        x = self.avgpool(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
     #   x=self.dropout(x)
        x = self.fc1(x)

        return x

def resgoust18(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Ghost2Net(Ghost2Module, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def resgoust34(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Ghost2Net(Ghost2Module, [3, 4, 6, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model
    
def resgoust50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Ghost2Net(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def resgoust101(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Ghost2Net(Bottleneck, [3, 4, 23, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model
    
def resgoust152(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Ghost2Net(Bottleneck, [3, 8, 36, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model



if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
