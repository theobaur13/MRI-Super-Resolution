import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.train.models.ESRGAN import Generator, Discriminator

def loop(epochs, dataloader, output_dir):
    generator = Generator().to("cuda")
    discriminator = Discriminator().to("cuda")

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    ### --- Pretraining the Generator --- ###
    pretrain_epochs = 5
    for epoch in tqdm(range(pretrain_epochs)):
        for lr, hr in tqdm(dataloader):
            lr, hr = lr.to("cuda"), hr.to("cuda")

            sr = generator(lr)
            loss = nn.L1Loss()(sr, hr)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()

        tqdm.write(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}], Loss: {loss.item():.4f}")

    ### --- Adversarial Training --- ###
    for epoch in tqdm(range(epochs)):
        count = 0
        for lr, hr in tqdm(dataloader):
            count += 1
            lr, hr = lr.to("cuda"), hr.to("cuda")

            valid = torch.ones((lr.size(0), 1), dtype=torch.float32, device="cuda")
            fake = torch.zeros((lr.size(0), 1), dtype=torch.float32, device="cuda")

            # Train Discriminator
            with torch.no_grad():
                sr = generator(lr)

            real_pred = discriminator(hr)
            fake_pred = discriminator(sr.detach())

            real_loss = adversarial_loss(real_pred, valid)
            fake_loss = adversarial_loss(fake_pred, fake)
            disc_loss = (real_loss + fake_loss) / 2

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            sr = generator(lr)
            real_pred = discriminator(hr)
            fake_pred = discriminator(sr)
            gen_adv_loss = adversarial_loss(fake_pred, valid)

            gen_content_loss = content_loss(sr, hr)
            gen_loss = gen_content_loss + 0.006 * gen_adv_loss
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            if count % 50 == 0:  # Print every 10 batches
                # Print losses
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Discriminator Loss: {disc_loss.item():.4f}")
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {gen_loss.item():.4f}")

                # Calculate metrics
                ssim_avg, psnr_avg = calculate_metrics(sr, hr)
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Training SSIM: {ssim_avg:.4f}, Training PSNR: {psnr_avg:.4f}")

                # Save generator state at each epoch
                torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_epoch_{epoch+1}.pth")
    
    # Save the model
    torch.save(generator.state_dict(), f"{output_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"{output_dir}/discriminator.pth")

def calculate_metrics(sr_batch, hr_batch):
    ssim_total = 0.0
    psnr_total = 0.0
    batch_size = sr_batch.size(0)

    for i in range(batch_size):
        sr_img = sr_batch[i].squeeze().detach().cpu().numpy()
        hr_img = hr_batch[i].squeeze().detach().cpu().numpy()

        ssim_total += ssim(hr_img, sr_img, data_range=1.0)
        psnr_total += psnr(hr_img, sr_img, data_range=1.0)

    return ssim_total / batch_size, psnr_total / batch_size