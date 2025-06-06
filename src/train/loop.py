import os
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.train.models.ESRGAN import Generator, Discriminator
from src.utils.readwrite import log_to_csv
from src.display.plot import plot_training_log

def loop(train_loader, val_loader, epochs, output_dir, resume=False):
    generator = Generator().to("cuda")
    discriminator = Discriminator().to("cuda")

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    pretrain_epochs = 4
    resume_pretrain_epoch = 0
    resume_epoch = 0

    if resume:
        models = os.listdir(output_dir)
        generators = [m for m in models if m.startswith("generator")]
        discriminators = [m for m in models if m.startswith("discriminator")]
        pretrain_gens = [m for m in models if m.startswith("pretrain")]

        if generators:
            latest_gen = sorted(generators, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            generator.load_state_dict(torch.load(os.path.join(output_dir, latest_gen)))
            resume_epoch = int(latest_gen.split('_')[-1].split('.')[0])
            resume_pretrain_epoch = pretrain_epochs
            tqdm.write(f"Resuming Generator from epoch {resume_epoch}")
        
        if discriminators:
            latest_disc = sorted(discriminators, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            discriminator.load_state_dict(torch.load(os.path.join(output_dir, latest_disc)))
            resume_pretrain_epoch = pretrain_epochs
            tqdm.write(f"Resuming Discriminator from epoch {resume_epoch}")

        if pretrain_gens:
            latest_pretrain = sorted(pretrain_gens, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            resume_pretrain_epoch = int(latest_pretrain.split('_')[-1].split('.')[0])
            tqdm.write(f"Resuming Pretraining from epoch {resume_pretrain_epoch}")
    
    log_file = os.path.join(output_dir, "history.csv")

    ### --- Pretraining the Generator --- ###
    for epoch in tqdm(range(resume_pretrain_epoch, pretrain_epochs)):
        for lr, hr in tqdm(train_loader):
            lr, hr = lr.to("cuda"), hr.to("cuda")

            sr = generator(lr)
            loss = nn.L1Loss()(sr, hr)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()

        torch.save(generator.state_dict(), f"{output_dir}/pretrain_epoch_{epoch+1}.pth")
        tqdm.write(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}], Loss: {loss.item():.4f}")

    ### --- Adversarial Training --- ###
    best_ssim = 0.0
    best_psnr = 0.0

    for epoch in tqdm(range(resume_epoch, epochs)):
        count = 0
        for lr, hr in tqdm(train_loader):
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

            if count % 100 == 0:  # Print every 100 batches
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Discriminator Loss: {disc_loss.item():.4f}")
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {gen_loss.item():.4f}")

                log_to_csv(log_file, {
                    "epoch": epoch + 1,
                    "batch": count,
                    "gen_loss": gen_loss.item(),
                    "disc_loss": disc_loss.item(),
                    "train_ssim": "",
                    "train_psnr": "",
                    "val_ssim": "",
                    "val_psnr": ""
                })

            if count % 1428 == 0:  # Print every 500 batches
                # Training Metrics
                ssim_avg, psnr_avg = calculate_metrics(sr, hr)
                tqdm.write(f"Epoch [{epoch+1}/{epochs}], Training SSIM: {ssim_avg:.4f}, Training PSNR: {psnr_avg:.4f}")

                # Validation Metrics
                with torch.no_grad():
                    val_ssim_total = 0.0
                    val_psnr_total = 0.0
                    for val_lr, val_hr in tqdm(val_loader):
                        val_lr, val_hr = val_lr.to("cuda"), val_hr.to("cuda")
                        val_sr = generator(val_lr)
                        ssim_val, psnr_val = calculate_metrics(val_sr, val_hr)
                        val_ssim_total += ssim_val
                        val_psnr_total += psnr_val

                    val_ssim_avg = val_ssim_total / len(val_loader)
                    val_psnr_avg = val_psnr_total / len(val_loader)
                    tqdm.write(f"Epoch [{epoch+1}/{epochs}], Validation SSIM: {val_ssim_avg:.4f}, Validation PSNR: {val_psnr_avg:.4f}")

                    if val_ssim_avg > best_ssim:
                        best_ssim = val_ssim_avg
                        torch.save(generator.state_dict(), f"{output_dir}/best_generator.pth")
                        torch.save(discriminator.state_dict(), f"{output_dir}/best_discriminator.pth")
                        tqdm.write(f"New best SSIM: {best_ssim:.4f}, model saved.")

                    if val_psnr_avg > best_psnr:
                        best_psnr = val_psnr_avg
                        torch.save(generator.state_dict(), f"{output_dir}/best_generator_psnr.pth")
                        torch.save(discriminator.state_dict(), f"{output_dir}/best_discriminator_psnr.pth")
                        tqdm.write(f"New best PSNR: {best_psnr:.4f}, model saved.")

                # Log metrics
                log_to_csv(log_file, {
                    "epoch": epoch + 1,
                    "batch": count,
                    "gen_loss": gen_loss.item(),
                    "disc_loss": disc_loss.item(),
                    "train_ssim": ssim_avg,
                    "train_psnr": psnr_avg,
                    "val_ssim": val_ssim_avg,
                    "val_psnr": val_psnr_avg
                })

                # Plot training log
                plot_training_log(log_file, output_dir=output_dir)

                # Save generator state
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