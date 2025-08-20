import os
import tensorflow as tf
import nibabel as nib
import numpy as np
import jax.numpy as jnp
import scipy.ndimage
from tqdm import tqdm
from scipy.stats import wilcoxon
from src.utils.inference import load_model
from src.utils.readwrite import read_nifti
from src.eval.helpers import get_grouped_slices, generate_SR_HR_LR_nifti_dir
from src.eval.DeepSeg.unet import construct_unet

def tumor(model_path, deepseg_path, lmdb_path, working_dir, set_type):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_tumor")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    model = load_model(model_path)
    generate_SR_HR_LR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path, set_type=set_type)

    segment_tumor(deepseg_path, input_dir, output_dir)

    dice_lr = calculate_dice(output_dir, "lr")
    dice_sr = calculate_dice(output_dir, "sr")

    # Perform Wilcoxon signed-rank test comparing LR and SR
    stat, p_value = wilcoxon(dice_lr, dice_sr)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")

def segment_tumor(deepseg_path, input_dir, output_dir):
    model = construct_unet(deepseg_path)

    for nifti_file in tqdm(os.listdir(input_dir)):
        if not nifti_file.endswith(".nii.gz"):
            continue

        if os.path.exists(os.path.join(output_dir, nifti_file)):
            tqdm.write(f"Skipping {nifti_file}, already processed.")
            continue

        # Load NIfTI file
        vol = nib.load(os.path.join(input_dir, nifti_file))
        vol_data = vol.get_fdata()
        vol_data = vol_data * 1000.0
        
        output_data = np.zeros_like(vol_data)
        for slice_idx in tqdm(range(vol_data.shape[2]), leave=False):
            img_data = vol_data[:, :, slice_idx]
            img_tensor = tf.convert_to_tensor(img_data, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            img_tensor = tf.expand_dims(img_tensor, axis=-1)
            img_tensor = tf.repeat(img_tensor, repeats=3, axis=-1)
            pred = model.predict(img_tensor, verbose=0)[0]
            pred_labels = np.argmax(pred, axis=-1)
            pred_image = pred_labels.reshape((240, 240))

            output_data[:, :, slice_idx] = pred_image

        output_img = nib.Nifti1Image(output_data, vol.affine, vol.header)
        nib.save(output_img, os.path.join(output_dir, nifti_file))

def calculate_dice(output_dir, comparison_type):
    dice_scores = []
    hr_files = sorted([f for f in os.listdir(output_dir) if "hr" in f and f.endswith(".nii.gz")])
    comparison_files = sorted([f for f in os.listdir(output_dir) if comparison_type in f and f.endswith(".nii.gz")])

    for hr_file, comparison_file in tqdm(zip(hr_files, comparison_files), total=len(hr_files), desc=f"Calculating DICE"):
        hr_path = os.path.join(output_dir, hr_file)
        comparison_path = os.path.join(output_dir, comparison_file)

        hr_vol = read_nifti(hr_path)
        comparison_vol = read_nifti(comparison_path)

        intersection = jnp.sum(hr_vol.get_fdata() * comparison_vol.get_fdata())
        dice = 2 * intersection / (jnp.sum(hr_vol.get_fdata()) + jnp.sum(comparison_vol.get_fdata()))
        dice_scores.append(dice)

    average_dice = jnp.mean(jnp.array(dice_scores))
    print(f"Average Dice for segmentation ({comparison_type} vs HR): {average_dice:.4f}")
    return dice_scores