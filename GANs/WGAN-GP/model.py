"""
Discriminator and Generator implementation from DCGAN and WGAN paper
reference: https://youtu.be/pG0QZ7OddX4
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super().__init__()

        # Input: N x img_channels x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),     # 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1),   # 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1),   # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),  # 1x1
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True), # InstanceNorm <--> LayerNorm
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, z_dim, img_channel, features_g):
        super().__init__()

        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1),  # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1),   # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1),   # 32x32
            # last conv layer doesnt use batch_norm
            nn.ConvTranspose2d(
                features_g*2, img_channel, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()   # [-1, 1]
        )

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    disc_out = disc(x)
    assert disc_out.shape == (N, 1, 1, 1)
    print("img shape:", x.shape)
    print("discriminator result:", disc_out.shape)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    gen_out = gen(z)
    assert gen_out.shape == (N, in_channels, H, W)
    print("z shape:", z.shape)
    print("generator result:", gen_out.shape)



if __name__ == "__main__":
    test()

