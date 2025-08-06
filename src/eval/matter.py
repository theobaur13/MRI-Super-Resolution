import os
import shutil
import subprocess
from tqdm import tqdm
import jax.numpy as jnp
from scipy.stats import wilcoxon
from src.utils.inference import load_model
from src.utils.readwrite import read_nifti
from src.eval.helpers import get_grouped_slices, generate_SR_HR_LR_nifti_dir

def matter(model_path, lmdb_path, flywheel_dir, working_dir, set_type):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_matter")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    model = load_model(model_path)
    generate_SR_HR_LR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path, set_type=set_type)

    segment_matter(flywheel_dir, input_dir, output_dir)
    for matter_type in ["csf", "wm", "gm"]:
        dice_lr = calculate_dice(output_dir, matter_type, "lr")
        dice_sr = calculate_dice(output_dir, matter_type, "sr")

        # Perform Wilcoxon signed-rank test
        stat, p_value = wilcoxon(dice_lr, dice_sr)
        print(f"Wilcoxon test statistic: {stat}, p-value: {p_value} for {matter_type} segmentation")

def segment_matter(flywheel_dir, input_dir, output_dir):
    # Set up directories
    flywheel_input_dir = os.path.join(flywheel_dir, "v0", "input", "nifti")
    flywheel_output_dir = os.path.join(flywheel_dir, "v0", "output")
    config_path = os.path.join(flywheel_dir, "v0", "config.json")

    # For each nifti in the input dir, copy into flywheel input dir
    for nifti in tqdm(os.listdir(input_dir)):
        if not nifti.endswith(".nii.gz"):
            continue

        # Check if the file already exists in the output directory
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

# Calculate Dice Similarity Coefficient between segmentation maps of SR and HR for each volume
def calculate_dice(output_dir, matter_type, comparison_type):
    dice_scores = []

    if matter_type == "csf":
        pve = 0
    elif matter_type == "wm":
        pve = 1
    elif matter_type == "gm":
        pve = 2

    # BraTS-GLI-00001-000-t2f_hr_fast_pve_0.nii.gz
    hr_files = [f for f in os.listdir(output_dir) if f.endswith(f"pve_{pve}.nii.gz") and "hr" in f]
    comparison_files = [f for f in os.listdir(output_dir) if f.endswith(f"pve_{pve}.nii.gz") and comparison_type in f]

    for hr_file, comparison_file in tqdm(zip(hr_files, comparison_files), total=len(hr_files), desc="Calculating Dice"):
        hr_path = os.path.join(output_dir, hr_file)
        comparison_path = os.path.join(output_dir, comparison_file)

        hr_vol = read_nifti(hr_path)
        comparison_vol = read_nifti(comparison_path)

        # Calculate Dice Similarity Coefficient
        intersection = jnp.sum(hr_vol.get_fdata() * comparison_vol.get_fdata())
        dice = 2 * intersection / (jnp.sum(hr_vol.get_fdata()) + jnp.sum(comparison_vol.get_fdata()))
        dice_scores.append(dice)

    average_dice = jnp.mean(jnp.array(dice_scores))
    print(f"Average Dice for {matter_type} segmentation ({comparison_type} vs HR): {average_dice:.4f}")
    return dice_scores