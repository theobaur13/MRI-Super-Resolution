import os
import shutil
import tensorflow as tf
import nibabel as nib
import numpy as np
import jax.numpy as jnp
import scipy.ndimage
from tqdm import tqdm
from scipy.stats import wilcoxon
from src.utils.inference import load_model
from src.utils.readwrite import read_nifti
from src.utils.paths import get_brats_paths
from src.eval.helpers import get_grouped_slices, generate_SR_HR_LR_nifti_dir

def tumor(model_path, latup_path, lmdb_path, working_dir, brats_dir, set_type):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_tumor")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    # model = load_model(model_path)
    # generate_SR_HR_LR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path, set_type=set_type)

    # segment_tumor(latup_path, input_dir, output_dir)

    for tumor_type in ["NEC", "EDE", "ENH"]:
        dice_lr = calculate_dice(output_dir, tumor_type, "lr")
        dice_sr = calculate_dice(output_dir, tumor_type, "sr")

        # Perform Wilcoxon signed-rank test comparing LR and SR
        stat, p_value = wilcoxon(dice_lr, dice_sr)
        print(f"Wilcoxon test statistic: {stat}, p-value: {p_value} for {tumor_type} segmentation (LR vs SR)")

def segment_tumor(latup_path, input_dir, output_dir):
    model = tf.keras.models.load_model(latup_path, compile=False)

    for nifti_file in tqdm(os.listdir(input_dir)):
        if not nifti_file.endswith(".nii.gz"):
            continue

        if os.path.exists(os.path.join(output_dir, nifti_file)):
            tqdm.write(f"Skipping {nifti_file}, already processed.")
            continue

        # Load NIfTI file
        img = nib.load(os.path.join(input_dir, nifti_file))
        img_data = img.get_fdata()

        # Reshape the image data to match the model input shape
        img_data = reshape_img(img_data)

        input_tensor = tf.convert_to_tensor(img_data[np.newaxis, ..., np.newaxis])
        output_tensor = model(input_tensor)

        output_data = output_tensor.numpy().squeeze()
        output_img = nib.Nifti1Image(output_data, img.affine, img.header)
        nib.save(output_img, os.path.join(output_dir, nifti_file))

# Reshape image of 240, 240, 120, to 128, 128, 128
def reshape_img(img_data):
    reshaped_data = scipy.ndimage.zoom(img_data, (128 / img_data.shape[0], 128 / img_data.shape[1], 128 / img_data.shape[2]), order=1)
    img_data = np.repeat(reshaped_data[..., np.newaxis], 3, axis=-1)
    return img_data

def calculate_dice(output_dir, tumor_type, comparison_type):
    dice_scores = []
    hr_files = sorted([f for f in os.listdir(output_dir) if "hr" in f and f.endswith(".nii.gz")])
    comparison_files = sorted([f for f in os.listdir(output_dir) if comparison_type in f and f.endswith(".nii.gz")])

    if tumor_type == "NEC":
        label_index = 1
    elif tumor_type == "EDE":
        label_index = 2
    elif tumor_type == "ENH":
        label_index = 3

    for hr_file, comparison_file in tqdm(zip(hr_files, comparison_files), total=len(hr_files), desc=f"Calculating DICE for {tumor_type}"):
        hr_path = os.path.join(output_dir, hr_file)
        comparison_path = os.path.join(output_dir, comparison_file)

        hr_vol = read_nifti(hr_path).get_fdata()  # Shape: (X, Y, Z, 4)
        comparison_vol = read_nifti(comparison_path).get_fdata()  # Shape: (X, Y, Z, 4)

        hr_label = hr_vol[..., label_index]
        comparison_label = comparison_vol[..., label_index]

        intersection = jnp.sum(hr_label * comparison_label)
        dice = 2 * intersection / (jnp.sum(hr_label) + jnp.sum(comparison_label))
        dice_scores.append(dice)

    average_dice = jnp.mean(jnp.array(dice_scores))
    print(f"Average Dice for {tumor_type} segmentation ({comparison_type} vs HR): {average_dice:.4f}")
    return dice_scores