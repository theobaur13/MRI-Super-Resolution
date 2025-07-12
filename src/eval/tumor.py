import os
import tensorflow as tf
import nibabel as nib
import numpy as np
import jax.numpy as jnp
import scipy.ndimage
from tqdm import tqdm
from src.utils.inference import load_model
from src.utils.readwrite import read_nifti
from src.eval.helpers import get_grouped_validation_slices, generate_SR_HR_nifti_dir

def tumor(model_path, latup_path, lmdb_path, working_dir):
    # Set up directories
    input_dir = os.path.join(working_dir, "input")
    output_dir = os.path.join(working_dir, "output_tumor")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    model = load_model(model_path)
    generate_SR_HR_nifti_dir(model, grouped_lr_paths, input_dir, lmdb_path)

    segment_tumor(latup_path, input_dir, output_dir)
    calculate_mae(output_dir)

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

def calculate_mae(output_dir, labels=["BG", "NEC", "EDE", "ENH"]):
    label_to_index = {"BG": 0, "NEC": 1, "EDE": 2, "ENH": 3}
    
    mae_per_label = {label: 0.0 for label in labels}
    count_per_label = {label: 0 for label in labels}

    hr_files = [f for f in os.listdir(output_dir) if "hr" in f and f.endswith(".nii.gz")]
    sr_files = [f for f in os.listdir(output_dir) if "sr" in f and f.endswith(".nii.gz")]

    hr_files.sort()
    sr_files.sort()

    for hr_file, sr_file in tqdm(zip(hr_files, sr_files), total=len(hr_files), desc="Calculating MAE"):
        hr_path = os.path.join(output_dir, hr_file)
        sr_path = os.path.join(output_dir, sr_file)

        hr_vol = read_nifti(hr_path).get_fdata()  # Shape: (X, Y, Z, 4)
        sr_vol = read_nifti(sr_path).get_fdata()  # Shape: (X, Y, Z, 4)

        for label in labels:
            idx = label_to_index[label]

            hr_label = hr_vol[..., idx]
            sr_label = sr_vol[..., idx]

            mae = jnp.mean(jnp.abs(hr_label - sr_label))
            mae_per_label[label] += mae
            count_per_label[label] += 1

    avg_mae = {}
    for label in labels:
        if count_per_label[label] > 0:
            avg = mae_per_label[label] / count_per_label[label]
            avg_mae[label] = float(avg)
            print(f"{label} MAE: {avg:.4f}")
        else:
            avg_mae[label] = float('nan')
            print(f"{label} MAE: NaN (no samples found)")

    return avg_mae