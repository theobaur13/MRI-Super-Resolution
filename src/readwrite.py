import os
import nibabel as nib
from nilearn import datasets, image
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import dicom2nifti
from src.utils import get_adni_paths

def read_nifti(file_path, brats=False):
    nifti = nib.load(file_path)
    nifti = nib.as_closest_canonical(nifti)

    # Step 1: Resample to MNI152 template
    # mni_template = datasets.load_mni152_template(resolution=1)
    # nifti = image.resample_to_img(nifti, mni_template, interpolation='continuous', force_resample=True, copy_header=True)

    # Step 2: Get data as JAX array
    volume = jnp.array(nifti.get_fdata())

    # Step 3: Optional axis reordering (e.g., BraTS)
    if brats:
        volume = jnp.transpose(volume, (2, 0, 1))

    # Step 4: Normalize intensity to [0, 1]
    volume = (volume - jnp.min(volume)) / (jnp.max(volume) - jnp.min(volume) + 1e-8)

    # Step 5: Re-wrap volume into a new NIfTI image (with original resampled affine)
    normalized_nifti = nib.Nifti1Image(np.array(volume), affine=nifti.affine)
    
    return normalized_nifti

def write_nifti(nifti, file_path):
    nib.save(nifti, file_path)

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