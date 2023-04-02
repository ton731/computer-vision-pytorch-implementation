import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),   # batch norm will cancel out the bias, so don't need bias
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64, 128, 256, 512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part for UNet
        for feature_channel in feature_channels:
            self.downs.append(DoubleConv(in_channels, feature_channel))
            in_channels = feature_channel
        
        # Up part for UNet
        for feature_channel in reversed(feature_channels):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature_channel*2, feature_channel, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature_channel*2, feature_channel))
        
        self.bottleneck = DoubleConv(feature_channels[-1], feature_channels[-1]*2)
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # down sampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # up sampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


def test():
    x = torch.randn((5, 3, 161, 161))
    model = UNet(in_channels=3, out_channels=1)
    output = model(x)
    assert output.shape == torch.Size((5, 1, 161, 161))
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test()
