import torch
from tqdm import tqdm
from src.eval.helpers import generate_SR_HR, get_grouped_slices

def mae(model_path, lmdb_path):
    set_type = "test"
    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    total_slices = sum(len(slices) for slices in grouped_lr_paths.values())

    total = 0.0
    count = 0

    for sr, hr in tqdm(generate_SR_HR(model_path, lmdb_path, set_type=set_type), total=total_slices, desc="Calculating MAE"):
        total += torch.mean(torch.abs(sr - hr)).item()
        count += 1

    mae = total / count if count > 0 else float('nan')
    print(f"Validation MAE: {mae}")
    return mae