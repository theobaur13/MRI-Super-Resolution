import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def calculate_mae(sr_batch, hr_batch):
    mae_total = 0.0
    batch_size = sr_batch.size(0)

    for i in range(batch_size):
        sr_img = sr_batch[i].squeeze().detach().cpu().numpy()
        hr_img = hr_batch[i].squeeze().detach().cpu().numpy()
        mae_total += np.mean(np.abs(sr_img - hr_img))

    return mae_total / batch_size