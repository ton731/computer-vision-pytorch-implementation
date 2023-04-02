"""
reference: https://www.youtube.com/watch?v=uQc4Fs7yx5I&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=19
"""

import torch
import torch.nn as nn



class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, residual_block_nums, image_channel, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(image_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.in_channels = 64
        self.layer1 = self._make_layer(
            block, residual_block_nums[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, residual_block_nums[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, residual_block_nums[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, residual_block_nums[3], intermediate_channels=512, stride=2
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

    def _make_layer(self, block, residual_block_num, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for example, 56x56 ->28x28 (stride=2), or
        # channels changes, we need to adapt the Identity (skip connection) so it will be 
        # able to be added to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(intermediate_channels * block.expansion)
            )
        
        # Add the first residual block which will adjust the input/identity
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # Adjust the new-coming input channel for the next residual blocks
        self.in_channels = intermediate_channels * block.expansion

        # Add the following resnet layer. For example first resnet layer: input 256
        # channel will be mapped to 64 as intermediate layer, then finally back to 
        # 256, hence no downsample is needed.
        for i in range(residual_block_num - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def ResNet50(image_channel=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], image_channel, num_classes)


def ResNet101(image_channel=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], image_channel, num_classes)
    

def ResNet152(image_channel=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], image_channel, num_classes)


def test():
    BATCH_SIZE = 5
    net = ResNet152(image_channel=3, num_classes=1000)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224))
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.shape)


if __name__ == "__main__":
    test()


