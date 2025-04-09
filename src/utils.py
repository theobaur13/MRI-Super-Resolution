import pandas as pd
import nibabel as nib
import jax.numpy as jnp
import SimpleITK as sitk
import torch
import pydicom
import os
from tqdm import tqdm
import re
from datetime import datetime

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def read_dicom(files):
    slices = []
    files.sort(key=lambda x: int(x.split("_")[-3]))  # Sort files by slice number
    for file in tqdm(files):
        if file.endswith(".dcm"):
            img = pydicom.dcmread(file)
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

def adni_dataframe(collapsed_path):
    data = []

    pattern = re.compile(
        r'ADNI_(\d+_S_\d+)_MR_([A-Za-z0-9_]+)_br_raw_(\d+)_([0-9]+)_S[0-9]+_(I\d+)'
    )

    for file in tqdm(os.listdir(collapsed_path)):
        match = pattern.match(file)
        if match:
            patient_id = match.group(1)
            description = match.group(2)
            timestamp_raw = match.group(3)
            slice_number = match.group(4)
            image_id = match.group(5)
            full_path = os.path.join(collapsed_path, file)

            data.append({
                'Patient ID': patient_id,
                'Description': description,
                'Timestamp': datetime.strptime(timestamp_raw, '%Y%m%d%H%M%S%f').date(),
                'Slice Number': int(slice_number),
                'Image ID': image_id,
                'File Path': full_path
            })

    return pd.DataFrame(data)

def get_adni_pair(df, index):
    df_1_5T = df[df["Description"] == "Axial_PD_T2_FSE_"]
    df_1_5T = df_1_5T.groupby(['Patient ID', 'Timestamp', 'Image ID'])
    patient_id, date, image_id = list(df_1_5T.groups.keys())[index][0], list(df_1_5T.groups.keys())[index][1], list(df_1_5T.groups.keys())[index][2]
    paths_1_5T = df_1_5T.get_group((patient_id, date, image_id))["File Path"].tolist()
    print(f"Patient ID: {patient_id}, Date: {date}, Image ID: {image_id}")

    df_3T = df[df["Description"] == "Double_TSE"]
    df_3T = df_3T.groupby(['Patient ID', 'Timestamp', 'Image ID'])
    _, _, image_id = list(df_3T.groups.keys())[index][0], list(df_3T.groups.keys())[index][1], list(df_3T.groups.keys())[index][2]
    paths_3T = df_3T.get_group((patient_id, date, image_id))["File Path"].tolist()
    print(f"Patient ID: {patient_id}, Date: {date}, Image ID: {image_id}")
    return paths_1_5T, paths_3T