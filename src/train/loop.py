import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.train.models.ESRGAN import Generator, Discriminator
from src.train.models.FSRCNN import FSRCNN
from src.utils.readwrite import log_to_csv
from src.utils.plot import plot_training_log
from src.utils.eval import calculate_metrics
from src.train.loss import CompositeLoss, gradient_penalty
from src.train.logging import resume_models, resume_log

def GAN_loop(train_loader, val_loader, epochs, pretrain_epochs, rrdb_count, output_dir, resume=False):
    # Initialize models
    generator = Generator(rrdb_count=rrdb_count).to("cuda")
    discriminator = Discriminator().to("cuda")

    # Initialize logging and resume settings
    resume_pretrain_epoch = 0
    resume_epoch = 0
    training_loss_file = os.path.join(output_dir, "training_loss.csv")
    ssim_file = os.path.join(output_dir, "ssim.csv")
    psnr_file = os.path.join(output_dir, "psnr.csv")
    individual_losses_file = os.path.join(output_dir, "individual_losses.csv")

    # Initialize optimizers and loss functions
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    content_loss = CompositeLoss(weights={
        "pixel": 0.3,
        "perceptual": 1.0,
        "edge": 0.0,
        "fourier": 0.0001,
        "style": 0.0
    })

    if resume:
        resume_pretrain_epoch, resume_epoch = resume_models(generator, discriminator, output_dir, pretrain_epochs)
        if resume_pretrain_epoch == pretrain_epochs:
            resume_log(training_loss_file, resume_epoch)
            resume_log(ssim_file, resume_epoch)
            resume_log(psnr_file, resume_epoch)

    ### --- Pretraining the Generator --- ###
    for epoch in tqdm(range(resume_pretrain_epoch + 1, pretrain_epochs + 1)):
        for lr, hr in tqdm(train_loader):
            lr, hr = lr.to("cuda"), hr.to("cuda")

            sr = generator(lr)
            loss = nn.L1Loss()(sr, hr)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()

        torch.save(generator.state_dict(), f"{output_dir}/pretrain_epoch_{epoch}.pth")
        tqdm.write(f"Pretrain Epoch [{epoch}/{pretrain_epochs}], Loss: {loss.item():.4f}")

    ### --- Adversarial Training --- ###
    best_ssim, best_psnr = 0.0, 0.0

    for epoch in tqdm(range(resume_epoch + 1, epochs)):
        count = 0
        for lr, hr in tqdm(train_loader):
            lr, hr = lr.to("cuda"), hr.to("cuda")

            # Train Discriminator
            sr = generator(lr).detach()

            real_pred = discriminator(hr)
            fake_pred = discriminator(sr)

            gp = gradient_penalty(discriminator, hr, sr, device="cuda", lambda_gp=10)

            disc_loss = fake_pred.mean() - real_pred.mean() + gp

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            sr = generator(lr)
            fake_pred = discriminator(sr)

            gen_adv_loss = -fake_pred.mean()

            gen_content_loss, losses = content_loss(sr, hr, logging=True)
            gen_loss = gen_content_loss + 0.001 * gen_adv_loss
            
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            count += 1
            if count % int(len(train_loader) / 50) == 0:  # Print every 10% of the set
                log_to_csv(training_loss_file, {
                    "epoch": epoch,
                    "batch": count,
                    "gen_loss": gen_loss.item(),
                    "disc_loss": disc_loss.item(),
                })

                individual_loss_data = {
                    "epoch": epoch,
                    "batch": count,
                }

                for key in ["pixel", "perceptual", "edge", "fourier", "style"]:
                    if key in losses:
                        individual_loss_data[f"{key}_loss"] = losses[key].item()

                log_to_csv(individual_losses_file, individual_loss_data)

            if count % int(len(train_loader) / 2) == 0:  # Print every 50% of the set
                # Training Metrics
                ssim_avg, psnr_avg = calculate_metrics(sr, hr)
                tqdm.write(f"Epoch [{epoch}/{epochs}], Training SSIM: {ssim_avg:.4f}, Training PSNR: {psnr_avg:.4f}")

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
                    tqdm.write(f"Epoch [{epoch}/{epochs}], Validation SSIM: {val_ssim_avg:.4f}, Validation PSNR: {val_psnr_avg:.4f}")

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
                log_to_csv(ssim_file, {
                    "epoch": epoch,
                    "batch": count,
                    "train_ssim": ssim_avg,
                    "val_ssim": val_ssim_avg
                })

                log_to_csv(psnr_file, {
                    "epoch": epoch,
                    "batch": count,
                    "train_psnr": psnr_avg,
                    "val_psnr": val_psnr_avg
                })

                # Plot training log
                # plot_training_log(training_loss_file, output_file=os.path.join(output_dir, "training_loss_plot.png"))
                # plot_training_log(ssim_file, output_file=os.path.join(output_dir, "ssim_plot.png"))
                # plot_training_log(psnr_file, output_file=os.path.join(output_dir, "psnr_plot.png"))
                # plot_training_log(individual_losses_file, output_file=os.path.join(output_dir, "individual_losses_plot.png"))

                # Save generator state
                torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch}.pth")
                torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_epoch_{epoch}.pth")
    
    # Save the model
    torch.save(generator.state_dict(), f"{output_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"{output_dir}/discriminator.pth")

def CNN_loop(train_loader, val_loader, epochs, output_dir, resume=False):
    # Initialize model
    model = FSRCNN().to("cuda")

    # Initialize logging and resume settings
    training_loss_file = os.path.join(output_dir, "training_loss.csv")
    ssim_file = os.path.join(output_dir, "ssim.csv")
    psnr_file = os.path.join(output_dir, "psnr.csv")

    resume_epoch = 0
    if resume:
        # Set model
        models = os.listdir(output_dir)

        def get_latest(name):
            filtered = [m for m in models if m.startswith(name)]
            return max(filtered, key=lambda x: int(x.split('_')[-1].split('.')[0])) if filtered else None
        
        latest_model = get_latest("fsrcnn_epoch")
        
        # Set resume epoch by checking existing models
        if latest_model:
            resume_epoch = int(latest_model.split('_')[-1].split('.')[0])
            model.load_state_dict(torch.load(os.path.join(output_dir, latest_model)))
            tqdm.write(f"Resuming from epoch {resume_epoch}")
        
        # Set csv files
        resume_log(training_loss_file, resume_epoch)
        resume_log(ssim_file, resume_epoch)
        resume_log(psnr_file, resume_epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CompositeLoss(
        weights={
            "pixel": 0.0,
            "perceptual": 1.0,
            "edge": 0.0,
            "fourier": 0.0,
            "style": 0.0
        }
    )

    ### --- Training Loop --- ###
    for epoch in tqdm(range(resume_epoch + 1, epochs + 1)):
        model.train()
        for lr, hr in tqdm(train_loader):
            lr, hr = lr.to("cuda"), hr.to("cuda")

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log_to_csv(training_loss_file, {
            #     "epoch": epoch, "loss": loss.item()
            # })

        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_ssim_total = 0.0
            val_psnr_total = 0.0
            for val_lr, val_hr in tqdm(val_loader):
                val_lr, val_hr = val_lr.to("cuda"), val_hr.to("cuda")
                val_sr = model(val_lr)
                ssim_val, psnr_val = calculate_metrics(val_sr, val_hr)
                val_ssim_total += ssim_val
                val_psnr_total += psnr_val

            val_ssim_avg = val_ssim_total / len(val_loader)
            val_psnr_avg = val_psnr_total / len(val_loader)

            log_to_csv(ssim_file, {
                "epoch": epoch,
                "val_ssim": val_ssim_avg
            })
            log_to_csv(psnr_file, {
                "epoch": epoch,
                "val_psnr": val_psnr_avg
            })

        # Save model state
        torch.save(model.state_dict(), f"{output_dir}/fsrcnn_epoch_{epoch}.pth")
    
    torch.save(model.state_dict(), f"{output_dir}/fsrcnn.pth")