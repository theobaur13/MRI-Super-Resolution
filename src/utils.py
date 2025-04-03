import pandas as pd
import nibabel as nib
import jax.numpy as jnp
import SimpleITK as sitk
import torch
import pydicom
import os
from tqdm import tqdm

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def read_metaimage(file_path):
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

def read_dicom(mri_dir):
    slices = []
    files = os.listdir(mri_dir)
    files.sort(key=lambda x: int(x.split("_")[-3]))  # Sort files by slice number
    for file in tqdm(files):
        if file.endswith(".dcm"):
            dicom_path = os.path.join(mri_dir, file)
            img = pydicom.dcmread(dicom_path)
            slices.append(img.pixel_array)

    return jnp.stack(slices, axis=0)

def convert_to_tensor(image_list, slice_axis):
    real_slices = []
    imag_slices = []

    for img in image_list:
        img_np = jnp.array(img)
        for i in range(img_np.shape[slice_axis]):
            slice_2d = img_np[i]
            real_slices.append(slice_2d.real)
            imag_slices.append(slice_2d.imag)
    
    real_slices = jnp.array(real_slices)
    imag_slices = jnp.array(imag_slices)

    real_tensor = torch.tensor(real_slices, dtype=torch.float32).unsqueeze(1)
    imag_tensor = torch.tensor(imag_slices, dtype=torch.float32).unsqueeze(1)
    return torch.cat((real_tensor, imag_tensor), dim=1)

def adni_search(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Field Strength'] = df['Imaging Protocol'].str.extract(r'(\d+\.\d+)').astype(float)

    # Filter Description must be "Axial PD/T2 FSE" or "Double_TSE"
    df = df[df['Description'].isin(["Axial PD/T2 FSE", "Double_TSE"])]

    # Group by Subject ID and Study Date, then filter for subjects that have both 1.5T and 3T
    valid_subjects = df.groupby(['Subject ID', 'Study Date'])['Field Strength'].nunique()
    valid_subjects = valid_subjects[valid_subjects > 1].index  # Keep only those with more than one unique field strength

    # Filter the original DataFrame
    filtered_df = df[df.set_index(['Subject ID', 'Study Date']).index.isin(valid_subjects)]

    # Save image ids to a txt file in comma-separated format (image_id_1, image_id_2, ...)
    image_ids = filtered_df['Image ID'].tolist()
    image_ids = [str(id) for id in image_ids]
    image_ids_str = ', '.join(image_ids)

    with open(output_path, 'w') as f:
        f.write(image_ids_str)
