from tqdm import tqdm
import torch
import lpips
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.eval.helpers import generate_SR_HR, get_grouped_slices

def metrics(model_path, lmdb_path, set_type="test"):
    loss_fn = lpips.LPIPS(net='alex').cuda()
    to_lpips_range = transforms.Normalize((0.5,), (0.5,))

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    count = 0
    mean_psnr = 0.0
    m2_psnr = 0.0
    mean_ssim = 0.0
    m2_ssim = 0.0
    mean_lpips = 0.0
    m2_lpips = 0.0

    with torch.no_grad():
        for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path, set_type=set_type), total=total_slices, desc="Calculating Metrics"):
            # --- LPIPS ---
            sr_lpips = to_lpips_range(sr).cuda()
            hr_lpips = to_lpips_range(hr).cuda()
            lpips_val = loss_fn(sr_lpips, hr_lpips).item()

            # --- PSNR ---
            sr_np = sr.squeeze().numpy()
            hr_np = hr.squeeze().numpy()
            psnr_val = calculate_psnr(hr_np, sr_np, data_range=1.0)

            # --- SSIM ---
            ssim_val = calculate_ssim(hr_np, sr_np, data_range=1.0)

            # Update statistics for LPIPS
            count += 1
            delta_lpips = lpips_val - mean_lpips
            mean_lpips += delta_lpips / count
            delta2_lpips = lpips_val - mean_lpips
            m2_lpips += delta_lpips * delta2_lpips

            # Update statistics for PSNR
            delta_psnr = psnr_val - mean_psnr
            mean_psnr += delta_psnr / count
            delta2_psnr = psnr_val - mean_psnr
            m2_psnr += delta_psnr * delta2_psnr
            
            # Update statistics for SSIM
            delta_ssim = ssim_val - mean_ssim
            mean_ssim += delta_ssim / count
            delta2_ssim = ssim_val - mean_ssim
            m2_ssim += delta_ssim * delta2_ssim

    # Calculate standard deviations
    if count < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    variance_lpips = m2_lpips / (count - 1)
    variance_psnr = m2_psnr / (count - 1)
    variance_ssim = m2_ssim / (count - 1)
    stdev_lpips = variance_lpips ** 0.5
    stdev_psnr = variance_psnr ** 0.5
    stdev_ssim = variance_ssim ** 0.5

    # Print results
    print(f"{set_type.capitalize()} LPIPS: {mean_lpips:.3f} ± {stdev_lpips:.3f}")
    print(f"{set_type.capitalize()} PSNR: {mean_psnr:.3f} ± {stdev_psnr:.3f}")
    print(f"{set_type.capitalize()} SSIM: {mean_ssim:.3f} ± {stdev_ssim:.3f}")
    return mean_lpips, stdev_lpips, mean_psnr, stdev_psnr, mean_ssim, stdev_ssim