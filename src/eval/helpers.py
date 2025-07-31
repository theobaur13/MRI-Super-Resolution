import lmdb
import torch
import os
import numpy as np
from tqdm import tqdm
from src.utils.readwrite import write_nifti
from src.utils.conversions import numpy_to_nifti
from src.utils.inference import load_model, run_model_on_slice

def group_slices(slices):
    grouped = {}
    for slice_key in slices:
        parts = slice_key.split('/')
        vol_id = parts[1]  # Extract volume ID
        if vol_id not in grouped:
            grouped[vol_id] = []
        grouped[vol_id].append(slice_key)

    # Order the slices by their index
    for vol_id, slice_keys in grouped.items():
        slice_keys.sort(key=lambda x: int(x.split("/")[-1]))
    return grouped

def get_LMDB_validate_paths(env):
    validate_prefix = b"validate/"
    print("Retrieving validation LR slice paths...")
    with env.begin() as txn:
        cursor = txn.cursor()
        validation_paths = []
        if cursor.set_range(validate_prefix):
            for key, _ in tqdm(cursor):
                if key.startswith(validate_prefix):
                    validation_paths.append(key.decode("utf-8"))
    return validation_paths

def get_grouped_validation_slices(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    validation_paths = get_LMDB_validate_paths(env)
    lr_paths = [p for p in validation_paths if "LR" in p]
    grouped_lr_paths = group_slices(lr_paths)
    return grouped_lr_paths

def generate_SR_HR(model_path, lmdb_path):
    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    print(f"Found {len(grouped_lr_paths)} validation volumes.")

    # Load the model
    model = load_model(model_path)
    model.eval()

    # Process each volume's LR slices
    with torch.no_grad():
        for volume, slice_keys in grouped_lr_paths.items():
            for lr_slice_key in slice_keys:
                slice_index = int(lr_slice_key.split("/")[-1])
                sr_slice, hr_slice, _ = run_model_on_slice(
                    model=model,
                    lmdb_path=lmdb_path,
                    vol_name=volume,
                    set_type="validate",
                    slice_index=slice_index,
                )

                # Yield as PyTorch tensors
                yield (
                    torch.tensor(sr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    torch.tensor(hr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )

def generate_SR_HR_LR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path):
    for volume, slices in tqdm(grouped_lr_paths.items(), desc="Processing LR slices"):
        sr_nifti_path = os.path.join(input_dir, f"{volume}_sr.nii.gz")
        hr_nifti_path = os.path.join(input_dir, f"{volume}_hr.nii.gz")
        lr_nifti_path = os.path.join(input_dir, f"{volume}_lr.nii.gz")

        # Skip if the NIfTI files already exist
        if os.path.exists(sr_nifti_path) and os.path.exists(hr_nifti_path) and os.path.exists(lr_nifti_path):
            print(f"Skipping {volume} (already processed)")
            continue
        
        sr_slices = []
        hr_slices = []
        lr_slices = []

        for lr_slice_key in slices:
            sr_slice, hr_slice, lr_slice = run_model_on_slice(
                model=model,
                lmdb_path=lmdb_path,
                vol_name=volume,
                set_type="validate",
                slice_index=int(lr_slice_key.split("/")[-1]),
            )

            sr_slices.append(sr_slice)
            hr_slices.append(hr_slice)
            lr_slices.append(lr_slice)

        # Stack the slices to create 3D volumes
        sr_volume = np.stack(sr_slices, axis=-1)
        hr_volume = np.stack(hr_slices, axis=-1)
        lr_volume = np.stack(lr_slices, axis=-1)

        # Any values close to 0 are set to 0
        sr_volume[sr_volume < 0.01] = 0

        # Create NIfTI images from the volumes
        sr_nifti = numpy_to_nifti(sr_volume)
        hr_nifti = numpy_to_nifti(hr_volume)
        lr_nifti = numpy_to_nifti(lr_volume)

        # Write the NIfTI images to output directory
        write_nifti(sr_nifti, sr_nifti_path)
        write_nifti(hr_nifti, hr_nifti_path)
        write_nifti(lr_nifti, lr_nifti_path)