import os
import nibabel as nib
import jax.numpy as jnp
import pydicom
from tqdm import tqdm
import SimpleITK as sitk
import dicom2nifti
from src.paths import get_adni_paths

def read_nifti(file_path, brats=False):
    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    image = jnp.array(img.get_fdata())  # Use get_fdata() for float64 and better precision
    
    if brats:
        # Convert from BraTS to Axial-Sagittal-Coronal
        image = jnp.transpose(image, (2, 0, 1))

    # image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image) + 1e-8)  # Added epsilon to avoid division by zero
    return image

def write_nifti(image, file_path):
    img = nib.Nifti1Image(image, affine=None)
    nib.save(img, file_path)

def convert_adni(adni_dir, output_dir):
    t1_5_paths, t3_paths = get_adni_paths(adni_dir)

    # Create 1.5T and 3T subdirectories
    os.makedirs(os.path.join(output_dir, "1.5T"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "3T"), exist_ok=True)

    def process(paths, target_dir):
        for dir in tqdm(paths):
            # Edit timestamps to reflect the directory name
            timestamp = dir.split("\\")[-2]
            timestamp = timestamp.replace(".0", "")
            timestamp = timestamp.replace("_", "")
            timestamp = timestamp.replace("-", "")
            contents = os.listdir(dir)
            
            old_name = contents[0]
            parts = old_name.split("_")
            parts[-4] = timestamp
            new_name = "_".join(parts)
            new_name = new_name.replace(".dcm", ".nii.gz")

            # Convert the volume to a NIfTI image
            dicom2nifti.dicom_series_to_nifti(dir, os.path.join(target_dir, new_name), reorient_nifti=True)

    process(t1_5_paths, os.path.join(output_dir, "1.5T"))
    process(t3_paths, os.path.join(output_dir, "3T"))