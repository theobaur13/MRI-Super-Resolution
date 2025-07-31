import nibabel as nib
import jax.numpy as jnp
import numpy as np
import os
import csv

def read_nifti(file_path, normalise=True):
    nifti = nib.load(file_path)
    nifti = nib.as_closest_canonical(nifti)
    volume = jnp.array(nifti.get_fdata())

    if normalise:
        volume = (volume - jnp.min(volume)) / (jnp.max(volume) - jnp.min(volume) + 1e-8)
    
    normalized_nifti = nib.Nifti1Image(np.array(volume), affine=nifti.affine)
    return normalized_nifti

def write_nifti(nifti, file_path):
    nib.save(nifti, file_path)

def log_to_csv(file_path, row, header=True):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if header and not file_exists:
            writer.writeheader()
        writer.writerow(row)