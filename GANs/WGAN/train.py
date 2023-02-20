"""
Training of DCGAN network on MNIST dataset with Discriminator and
Generator imported from models.py using Wasserstein distance and WGAN settings.
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
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
# IMAGE_CHANNELS = 1
IMAGE_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 30
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        )
    ]
)

# if train MNIST, need to change the img_channel to 1.
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

dataset = datasets.ImageFolder(root="dataset/celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
critic = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

optimizer_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
optimizer_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter("logs/MNIST/real")
# writer_fake = SummaryWriter("logs/MNIST/fake")
writer_real = SummaryWriter("logs/celeb/real")
writer_fake = SummaryWriter("logs/celeb/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        
        ### Train Discriminator: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### Train Generator: min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # tensorboard
        if batch_idx % 500 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], Batch: {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples 
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)

            step += 1


