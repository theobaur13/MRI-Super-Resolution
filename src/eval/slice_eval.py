import torch
from tqdm import tqdm
from torchvision import transforms
import lpips
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.eval.helpers import get_grouped_slices
import matplotlib.pyplot as plt
from src.eval.helpers import generate_SR_HR
from collections import defaultdict
import numpy as np

def slice_eval(model_path, lmdb_path, set_type="test"):
    loss_fn = lpips.LPIPS(net='alex').cuda()
    to_lpips_range = transforms.Normalize((0.5,), (0.5,))

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    # slice_indices = []
    # lpips_vals = []
    # psnr_vals = []
    # ssim_vals = []

    metrics_by_slice = defaultdict(lambda: {'lpips': [], 'psnr': [], 'ssim': []})

    with torch.no_grad():
        for sr, hr, slice_idx in tqdm(
            generate_SR_HR(model_path, lmdb_path, set_type=set_type, return_index=True),
            total=total_slices,
            desc="Calculating Metrics"
        ):
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

            # Save metrics
            metrics_by_slice[slice_idx]['lpips'].append(lpips_val)
            metrics_by_slice[slice_idx]['psnr'].append(psnr_val)
            metrics_by_slice[slice_idx]['ssim'].append(ssim_val)

    # --- Compute average per slice index ---
    sorted_slice_indices = sorted(metrics_by_slice.keys())
    avg_lpips = [np.mean(metrics_by_slice[idx]['lpips']) for idx in sorted_slice_indices]
    avg_psnr = [np.mean(metrics_by_slice[idx]['psnr']) for idx in sorted_slice_indices]
    avg_ssim = [np.mean(metrics_by_slice[idx]['ssim']) for idx in sorted_slice_indices]

    # --- Plotting ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(sorted_slice_indices, avg_lpips, label="Avg LPIPS", color="blue")
    plt.xlabel("Slice Index")
    plt.ylabel("LPIPS")
    plt.title("Average LPIPS per slice")

    plt.subplot(1, 3, 2)
    plt.scatter(sorted_slice_indices, avg_psnr, label="Avg PSNR", color="green")
    plt.xlabel("Slice Index")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR over slices")

    plt.subplot(1, 3, 3)
    plt.scatter(sorted_slice_indices, avg_ssim, label="Avg SSIM", color="orange")
    plt.xlabel("Slice Index")
    plt.ylabel("SSIM")
    plt.title("SSIM over slices")

    plt.tight_layout()
    plt.show()