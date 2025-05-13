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