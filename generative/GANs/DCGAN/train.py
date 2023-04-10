"""
Training of DCGAN network on MNIST dataset with Discriminator and
Generator imported from models.py
CelebA dataset: https://www.kaggle.com/datasets/504743cb487a5aed565ce14238c6343b7d650ffd28c071f03f2fd9b25819e6c9
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from distutils.version import LooseVersion  # to remove tensorboard error
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
# IMAGE_CHANNELS = 1
IMAGE_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 30
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        )
    ]
)

# dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root="../dataset/celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
disc = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

optimizer_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter("logs/MNIST/real")
# writer_fake = SummaryWriter("logs/MNIST/fake")
writer_real = SummaryWriter("logs/celeb/real")
writer_fake = SummaryWriter("logs/celeb/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(x)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        fake = gen(noise)
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        ### Train Generator: min log(1 - D(G(x))) <--> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # tensorboard
        # if batch_idx % 100 == 0:
        if batch_idx % 500 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], Batch: {batch_idx}/{len(loader)} \
                    Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples 
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)

            step += 1


