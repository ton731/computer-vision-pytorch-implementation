"""
refrence: https://youtu.be/OljTVUVzPpM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from distutils.version import LooseVersion  # to remove tensorboard error
from torch.utils.tensorboard import SummaryWriter



# Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3. Different learning rate (is there a better one)?
# 4. Change architecture to a CNN




class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # to make output between -1 ~ 1, which is same as the later img normalization
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64  # 128, 256
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer_disc = optim.Adam(disc.parameters(), lr=lr)
optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizer_disc.zero_grad()
        loss_D.backward()
        optimizer_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <--> max log(D(G(z)))
        output  = disc(fake).view(-1)
        loss_G = criterion(output, torch.ones_like(output))
        optimizer_gen.zero_grad()
        loss_G.backward()
        optimizer_gen.step()

        # tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist real Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist real Images", img_grid_real, global_step=step
                )

                step += 1



