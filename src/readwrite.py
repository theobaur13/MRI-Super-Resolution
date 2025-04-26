import os
import shutil
import nibabel as nib
import jax.numpy as jnp
import pydicom
from tqdm import tqdm
from src.paths import get_adni_paths
import numpy as np

def read_nifti(file_path, brats=False):
    img = nib.load(file_path)
    image = np.asanyarray(img.dataobj)
    if brats:
        # In BraTS Axis 0: Sagittal, Axis 1: Coronal, Axis 2: Axial
        # Convert Axis 0: Axial, Axis 1: Sagittal, Axis 2: Coronal
        image = np.transpose(image, (2, 0, 1))

    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image

def write_nifti(image, file_path):
    img = nib.Nifti1Image(image, affine=None)
    nib.save(img, file_path)

def read_dicom(files, flip=False):
    slices = []
    files.sort(key=lambda x: int(x.split("_")[-3]))  # Sort files by slice number
    for file in files:
        if file.endswith(".dcm"):
            img = pydicom.dcmread(file)
            slices.append(img.pixel_array)

    image = jnp.stack(slices, axis=0)

    if flip:
        image = jnp.flip(image, axis=0)

    normalized_image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))
    return normalized_image

def collapse_adni(adni_dir, output_dir):
    t1_5_paths, t3_paths = get_adni_paths(adni_dir, limit=100000)
    paths = t1_5_paths + t3_paths

    os.makedirs(output_dir, exist_ok=True)

    for dir in tqdm(paths):
        contents = os.listdir(dir)
        timestamp = dir.split("\\")[-2]
        timestamp = timestamp.replace(".0", "")
        timestamp = timestamp.replace("_", "")
        timestamp = timestamp.replace("-", "")

        for file in contents:
            if file.endswith(".dcm"):
                old_name = file
                parts = old_name.split("_")
                parts[-4] = timestamp
                new_name = "_".join(parts)
                shutil.copy(os.path.join(dir, file), os.path.join(output_dir, new_name))