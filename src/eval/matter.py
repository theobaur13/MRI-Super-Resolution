import os
import shutil
import subprocess
from tqdm import tqdm
import jax.numpy as jnp
from src.utils.inference import load_model
from src.utils.readwrite import read_nifti
from src.eval.helpers import get_grouped_validation_slices, generate_SR_HR_nifti_dir

def matter(model_path, lmdb_path, flywheel_dir, working_dir):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_matter")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    model = load_model(model_path)
    generate_SR_HR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path)

    segment_matter(flywheel_dir, input_dir, output_dir)
    calculate_mae(output_dir, "gm")

def segment_matter(flywheel_dir, input_dir, output_dir):
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
        shutil.copy(src, dest)

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

# Calculate Mean Absolute Error between segmentation maps of SR and HR for each volume
def calculate_mae(output_dir, matter_type):
    total_mae = 0
    count = 0

    if matter_type == "csf":
        pve = 0
    elif matter_type == "wm":
        pve = 1
    elif matter_type == "gm":
        pve = 2

    # BraTS-GLI-00001-000-t2f_hr_fast_pve_0.nii.gz
    hr_files = [f for f in os.listdir(output_dir) if f.endswith(f"pve_{pve}.nii.gz") and "hr" in f]
    sr_files = [f for f in os.listdir(output_dir) if f.endswith(f"pve_{pve}.nii.gz") and "sr" in f]

    for hr_file, sr_file in tqdm(zip(hr_files, sr_files)):
        hr_path = os.path.join(output_dir, hr_file)
        sr_path = os.path.join(output_dir, sr_file)

        hr_vol = read_nifti(hr_path)
        sr_vol = read_nifti(sr_path)

        # Calculate Mean Absolute Error
        mae = jnp.mean(jnp.abs(hr_vol.get_fdata() - sr_vol.get_fdata()))

        total_mae += mae
        count += 1

    
    average_mae = total_mae / count
    print(f"Average MAE for {matter_type} segmentation: {average_mae:.4f}")
    return average_mae