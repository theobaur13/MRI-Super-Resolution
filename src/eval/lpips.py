import torch
import torchvision.transforms as transforms
import lpips  
from tqdm import tqdm
from src.eval.helpers import generate_SR_HR, get_grouped_validation_slices

def LPIPS(model_path, lmdb_path):
    loss_fn = lpips.LPIPS(net='alex').cuda()  
    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    total = 0.0
    count = 0

    to_lpips_range = transforms.Normalize((0.5,), (0.5,))

    with torch.no_grad():
        for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path), total=total_slices, desc="Calculating LPIPS"):
            sr = to_lpips_range(sr).cuda()
            hr = to_lpips_range(hr).cuda()

            # Calculate LPIPS loss
            loss = loss_fn(sr, hr)
            total += loss.item()
            count += 1

    lpips_value = total / count if count > 0 else float('nan')
    print(f"Validation LPIPS: {lpips_value:.6f}")