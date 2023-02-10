"""
reference: https://www.youtube.com/watch?v=ACmuBbuXn20&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=17
"""

import torch
import torch.nn as nn


# architecture ('M' is the max pooling)
VGG16_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# then flatten and 4096 x 4096 x 1000 linear layers

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16_layers)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),   # this is optional, batch_norm is not invented at that time, but it improves performance
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x




if __name__ == "__main__":
    img = torch.randn(8, 3, 224, 224)
    model = VGG(in_channels=3, num_classes=1000)
    output = model(img)
    print(output.shape)


