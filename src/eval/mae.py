import lmdb
from tqdm import tqdm
import torch
from src.utils.eval import calculate_mae
from src.eval.helpers import group_slices, get_LMDB_validate_paths
from src.utils.inference import load_model, run_model_on_slice

def mae(model_path, lmdb_path):
    # Get all validation LR slice keys
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    validation_paths = get_LMDB_validate_paths(env)

    lr_paths = [p for p in validation_paths if "LR" in p]
    grouped_lr_paths = group_slices(lr_paths)
    print(f"Found {len(grouped_lr_paths)} validation volumes.")

    # Run model and collect SR and HR batches
    all_sr_slices = []
    all_hr_slices = []

    # Load the model
    model = load_model(model_path)

    # Process each volume's LR slices
    for volume, slice_keys in tqdm(grouped_lr_paths.items(), desc="Running model on validation set"):
        for lr_slice_key in slice_keys:
            slice_index = int(lr_slice_key.split("/")[-1])
            sr_slice, hr_slice, _ = run_model_on_slice(
                model=model,
                lmdb_path=lmdb_path,
                vol_name=volume,
                set_type="validate",
                slice_index=slice_index,
            )

            # Convert to PyTorch tensors and accumulate
            sr_tensor = torch.tensor(sr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
            hr_tensor = torch.tensor(hr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

            all_sr_slices.append(sr_tensor)
            all_hr_slices.append(hr_tensor)

    if not all_sr_slices:
        raise ValueError("No slices processed — is your LMDB path and structure correct?")

    # Stack all tensors into batches
    sr_batch = torch.cat(all_sr_slices, dim=0)  # shape: [N, 1, H, W]
    hr_batch = torch.cat(all_hr_slices, dim=0)  # shape: [N, 1, H, W]

    # Compute MAE
    mae_value = calculate_mae(sr_batch, hr_batch)
    print(f"Validation MAE: {mae_value:.6f}")
    return mae_value