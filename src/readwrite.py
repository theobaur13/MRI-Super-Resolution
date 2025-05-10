import os
import shutil
import nibabel as nib
import jax.numpy as jnp
import pydicom
from tqdm import tqdm
import SimpleITK as sitk
from src.paths import get_adni_paths

def read_nifti(file_path, brats=False):
    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    image = jnp.array(img.get_fdata())  # Use get_fdata() for float64 and better precision
    
    if brats:
        # Convert from BraTS to Axial-Sagittal-Coronal
        image = jnp.transpose(image, (2, 0, 1))

    image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image) + 1e-8)  # Added epsilon to avoid division by zero
    return image

def write_nifti(image, file_path):
    img = nib.Nifti1Image(image, affine=None)
    nib.save(img, file_path)

def read_dicom(files, reorient=True):
    slices = []
    files.sort(key=lambda x: int(x.split("_")[-3]))  # Sort files by slice number
    for file in files:
        if file.endswith(".dcm"):
            img = pydicom.dcmread(file)
            slices.append(img.pixel_array)

    image = jnp.stack(slices, axis=0)

    if reorient:
        # Reorient the 3D array using SimpleITK
        # (a) Read the whole DICOM series properly
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        sitk_image = reader.Execute()
        
        # (b) Reorient to RAS+ (Right-Anterior-Superior) orientation
        sitk_canonical = sitk.DICOMOrient(sitk_image, 'ARS')
        
        # (c) Convert back to a numpy array (SimpleITK uses z,y,x ordering)
        np_image = sitk.GetArrayFromImage(sitk_canonical)  # shape (slices, height, width)
        
        # (d) Convert to jax.numpy
        image = jnp.array(np_image)

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