from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.eval.helpers import generate_SR_HR, get_grouped_validation_slices

def psnr(model_path, lmdb_path):
    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    total_psnr = 0.0
    count = 0

    # Generate Super-Resolution (SR) and High-Resolution (HR) images
    for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path), total=total_slices, desc="Calculating PSNR"):
        sr_img = sr.squeeze().numpy()
        hr_img = hr.squeeze().numpy()

        total_psnr += calculate_psnr(hr_img, sr_img, data_range=1.0)
        count += 1

    avg_psnr = total_psnr / count if count > 0 else float('nan')

    print(f"Validation PSNR: {avg_psnr:.6f}")
    return avg_psnr