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

def tumor(model_path, latup_path, lmdb_path, working_dir, brats_dir):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_tumor")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    set_type = "test"

    grouped_lr_paths = get_grouped_slices(lmdb_path, set_type=set_type)
    model = load_model(model_path)
    generate_SR_HR_LR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path, set_type=set_type)

    segment_tumor(latup_path, input_dir, output_dir)

    # Copy ground truth files to output directory
    _, _, gt_paths = get_brats_paths(brats_dir, seq="t2f", dataset="BraSyn")
    for gt_path in gt_paths:
        if not os.path.exists(gt_path):
            continue
        gt_file = os.path.basename(gt_path)
        shutil.copy(gt_path, os.path.join(output_dir, gt_file))

    dice_lr = calculate_dice(output_dir, "lr")
    dice_sr = calculate_dice(output_dir, "sr")
    dice_hr = calculate_dice(output_dir, "hr")

    # Perform Wilcoxon signed-rank test comparing LR and SR
    stat, p_value = wilcoxon(dice_lr["NEC"], dice_sr["NEC"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    stat, p_value = wilcoxon(dice_lr["EDE"], dice_sr["EDE"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    stat, p_value = wilcoxon(dice_lr["ENH"], dice_sr["ENH"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")

    # Perform Wilcoxon signed-rank test comparing SR and HR
    stat, p_value = wilcoxon(dice_sr["NEC"], dice_hr["NEC"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    stat, p_value = wilcoxon(dice_sr["EDE"], dice_hr["EDE"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    stat, p_value = wilcoxon(dice_sr["ENH"], dice_hr["ENH"])
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")

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

def calculate_dice(output_dir, comparison_type, labels=["BG", "NEC", "EDE", "ENH"]):
    label_to_index = {"BG": 0, "NEC": 1, "EDE": 2, "ENH": 3}

    dice_scores_per_label = {label: [] for label in labels}

    gt_files = [f for f in os.listdir(output_dir) if "gt" in f and f.endswith(".nii.gz")]
    comparison_files = [f for f in os.listdir(output_dir) if comparison_type in f and f.endswith(".nii.gz")]

    gt_files.sort()
    comparison_files.sort()

    for gt_file, comparison_file in tqdm(zip(gt_files, comparison_files), total=len(gt_files), desc="Calculating DICE"):
        gt_path = os.path.join(output_dir, gt_file)
        comparison_path = os.path.join(output_dir, comparison_file)

        gt_vol = read_nifti(gt_path).get_fdata()  # Shape: (X, Y, Z, 4)
        comparison_vol = read_nifti(comparison_path).get_fdata()  # Shape: (X, Y, Z, 4)

        for label in labels:
            idx = label_to_index[label]

            gt_label = gt_vol[..., idx]
            comparison_label = comparison_vol[..., idx]

            dice = calculate_dice_coefficient(gt_label, comparison_label)
            dice_scores_per_label[label].append(dice)

    avg_dice = {}
    for label in labels:
        scores = dice_scores_per_label[label]
        if scores:
            avg_dice[label] = np.mean(scores)
            print(f"Average DICE for {label}: {avg_dice[label]:.4f}")

    return dice_scores_per_label

def calculate_dice_coefficient(vol1, vol2, eps=1e-7):
    """Calculate Dice coefficient between two binary volumes."""
    vol1 = jnp.asarray(vol1)
    vol2 = jnp.asarray(vol2)
    intersection = jnp.sum(vol1 * vol2)
    union = jnp.sum(vol1) + jnp.sum(vol2)
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice)