from tqdm import tqdm
from skimage.metrics import structural_similarity as calculate_ssim
from src.eval.helpers import generate_SR_HR, get_grouped_slices

def ssim(model_path, lmdb_path):
    set_type = "test"
    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())
    
    total_ssim = 0.0
    count = 0

    for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path, set_type=set_type), total=total_slices, desc="Calculating SSIM"):
        sr_img = sr.squeeze().numpy()
        hr_img = hr.squeeze().numpy()

        total_ssim += calculate_ssim(hr_img, sr_img, data_range=1.0)
        count += 1

    avg_ssim = total_ssim / count if count > 0 else float('nan')

    print(f"Validation SSIM: {avg_ssim:.6f}")
    return avg_ssim

