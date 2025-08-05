import torch
import torchvision.transforms as transforms
import lpips  
from tqdm import tqdm
from src.eval.helpers import generate_SR_HR, get_grouped_slices

def LPIPS(model_path, lmdb_path, set_type="test"):
    loss_fn = lpips.LPIPS(net='alex').cuda()

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    count = 0
    mean = 0.0
    m2 = 0.0

    to_lpips_range = transforms.Normalize((0.5,), (0.5,))

    with torch.no_grad():
        for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path, set_type=set_type), total=total_slices, desc="Calculating LPIPS"):
            sr = to_lpips_range(sr).cuda()
            hr = to_lpips_range(hr).cuda()

            # Calculate LPIPS loss
            loss = loss_fn(sr, hr)
            lpips_val = loss.item()
            count += 1
            delta = lpips_val - mean
            mean += delta / count
            delta2 = lpips_val - mean
            m2 += delta * delta2

    if count < 2:
        return 0.0
    variance = m2 / (count - 1)
    stddev = variance ** 0.5
    print(f"{set_type.capitalize()} LPIPS: {mean:.3f} ± {stddev:.3f}")
    return mean, stddev