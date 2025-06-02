import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.ESRGAN import Generator, Discriminator

def loop(epochs, dataloader, batch_size):
    generator = Generator().to("cuda")
    discriminator = Discriminator().to("cuda")

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    for epoch in tqdm(range(epochs)):
        for lr, hr in dataloader:
            lr, hr = lr.to("cuda"), hr.to("cuda")