import os
import shutil
import subprocess
import lmdb
import numpy as np
from tqdm import tqdm
from src.utils.inference import load_model, run_model_on_slice
from src.utils.conversions import numpy_to_nifti
from src.utils.readwrite import write_nifti
from src.eval.helpers import group_slices, get_LMDB_validate_paths

def matter(model_path, lmdb_path, flywheel_dir, working_dir):
    # Retrieve all "validate" slices from the LMDB dataset
    env = lmdb.open(lmdb_path, readonly=True)
    validation_paths = get_LMDB_validate_paths(env)

    print(f"Found {len(validation_paths)} validate slices in LMDB dataset.")

    # Seperate the slices into HR and LR
    lr_paths = [s for s in validation_paths if "LR" in s]

    # Group the slices by volume ID (structure: validate/vol_id/LR/slice_index)
    grouped_lr_paths = group_slices(lr_paths)

    input_dir = os.path.join(working_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Load the model
    model = load_model(model_path)

    # Run LR slices through the model
    print(f"Processing {len(grouped_lr_paths)} volumes with LR slices...")
    for volume, slices in tqdm(grouped_lr_paths.items(), desc="Processing LR slices"):
        sr_nifti_path = os.path.join(input_dir, f"{volume}_sr.nii.gz")
        hr_nifti_path = os.path.join(input_dir, f"{volume}_hr.nii.gz")

        # Skip if the NIfTI files already exist
        if os.path.exists(sr_nifti_path) and os.path.exists(hr_nifti_path):
            print(f"Skipping {volume} (already processed)")
            continue
        
        sr_slices = []
        hr_slices = []

        for lr_slice_key in slices:
            sr_slice, hr_slice, _ = run_model_on_slice(
                model=model,
                lmdb_path=lmdb_path,
                vol_name=volume,
                set_type="validate",
                slice_index=int(lr_slice_key.split("/")[-1]),
            )

            sr_slices.append(sr_slice)
            hr_slices.append(hr_slice)

        # Stack the slices to create 3D volumes
        sr_volume = np.stack(sr_slices, axis=-1)
        hr_volume = np.stack(hr_slices, axis=-1)

        # Create NIfTI images from the volumes
        sr_nifti = numpy_to_nifti(sr_volume)
        hr_nifti = numpy_to_nifti(hr_volume)

        # Write the NIfTI images to output directory
        write_nifti(sr_nifti, sr_nifti_path)
        write_nifti(hr_nifti, hr_nifti_path)

    # Run HR and SR volumes through the FSL FAST segmentation
    output_dir = os.path.join(working_dir, "output")
    segment(flywheel_dir, input_dir, output_dir)

    # Calculate MSE for WM, GM, CSF for SR and HR
    pass

def segment(flywheel_dir, input_dir, output_dir):
    # Set up directories
    flywheel_input_dir = os.path.join(flywheel_dir, "v0", "input", "nifti")
    flywheel_output_dir = os.path.join(flywheel_dir, "v0", "output")
    config_path = os.path.join(flywheel_dir, "v0", "config.json")

    # For each nifti in the input dir, copy into flywheel input dir
    for nifti in tqdm(os.listdir(input_dir)):
        if not nifti.endswith(".nii.gz"):
            continue

        # Check if the file already exists in the flywheel output directory
        output_files = set(os.listdir(output_dir))
        prefix = nifti.split(".")[0]

        if any(prefix in file for file in output_files):
            print(f"Skipping {nifti} as it is already segmented.")
            continue

        src = os.path.join(input_dir, nifti)
        dest = os.path.join(flywheel_input_dir, nifti)
        shutil.move(src, dest)

        # Run segmentation using FSL FAST
        subprocess.run([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{config_path}:/flywheel/v0/config.json",
            "-v", f"{flywheel_input_dir}:/flywheel/v0/input/nifti",
            "-v", f"{flywheel_output_dir}:/flywheel/v0/output",
            "scitran/fsl-fast",
            "-t", "2"
        ])

        # Move the flywheel output files to another directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in os.listdir(flywheel_output_dir):
            if file.endswith(".nii.gz"):
                src = os.path.join(flywheel_output_dir, file)
                dest = os.path.join(output_dir, file)
                shutil.move(src, dest)

        # Clean up the flywheel input directory
        for file in os.listdir(flywheel_input_dir):
            if file.endswith(".nii.gz"):
                os.remove(os.path.join(flywheel_input_dir, file))