
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


class res(nn.Module):
    expansion = 1
    def __init__(self, inp,oup,  stride=1, relu=True,downsample=False):

        super(res, self).__init__()
        self.midp = oup
        self.stride=stride


        self.conv1 = nn.Conv2d(inp, self.midp, kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.midp)

        self.conv2 = nn.Conv2d(self.midp, oup, kernel_size=3,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

        # shortcut
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
       # print(x.shape)
      #  print(out.shape)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inp,oup,  stride=1, relu=True,downsample=False):

        super(Bottleneck, self).__init__()
        self.midp = oup
        self.stride=stride


        self.conv1 = nn.Conv2d(inp, self.midp, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.midp)

        self.conv2 = nn.Conv2d(self.midp, oup, kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.conv3 = nn.Conv2d(self.midp, oup * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup* self.expansion)

        # shortcut
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup* self.expansion, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(oup* self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        print(x.shape)
        print(out.shape)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
        

class ResNet(nn.Module):

    def __init__(self, block, layers,   num_classes=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
       #                        bias=False)
     #   self.bn2 = nn.BatchNorm2d(64)


        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion, 100)
        self.fc1 = nn.Linear(100, num_classes)


    def _make_layer(self, block,planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes,  planes, stride=stride,downsample=True))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,  planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
     #   x = self.conv2(x)
      #  x = self.bn2(x)
       # x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       # print(x.shape)

        x = self.avgpool(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)

        return x



def resnet18(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(res, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(res, [3, 4, 6, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model
    
def resnet50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model
    
def resnet152(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],**kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model

if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
