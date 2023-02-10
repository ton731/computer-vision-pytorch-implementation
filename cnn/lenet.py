"""
reference: https://www.youtube.com/watch?v=fcOW-Zyb5Bo&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=16
LeNet-5 architecture:
1x32x32 input -> (5x5),s=1,p=0 -> (avg pool),s=2,p=0 -> (5x5),s=1,p=0 -> (avg pool),s=2,p=0
    -> (5x5) to 120 channels -> (linear),120x84 -> (linear),84x10
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))    # 120 x 1 x 1
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x



if __name__ == "__main__":
    img = torch.randn(64, 1, 32, 32)
    model = LeNet()
    output = model(img)
    print(output.shape)
