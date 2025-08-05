from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.eval.helpers import generate_SR_HR, get_grouped_slices

def psnr(model_path, lmdb_path, set_type="test"):
    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    count = 0
    mean = 0.0
    m2 = 0.0

    # Generate Super-Resolution (SR) and High-Resolution (HR) images
    for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path, set_type=set_type), total=total_slices, desc="Calculating PSNR"):
        sr_img = sr.squeeze().numpy()
        hr_img = hr.squeeze().numpy()

        psnr_val = calculate_psnr(hr_img, sr_img, data_range=1.0)
        
        count += 1
        delta = psnr_val - mean
        mean += delta / count
        delta2 = psnr_val - mean
        m2 += delta * delta2

    if count < 2:
        return 0.0
    variance = m2 / (count - 1)
    stddev = variance ** 0.5

    print(f"{set_type.capitalize()} PSNR: {mean:.3f} ± {stddev:.3f}")
    return mean, stddev