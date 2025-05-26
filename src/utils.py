import os
import re
from datetime import datetime
import jax.numpy as jnp
from tqdm import tqdm

def robust_max(image, axis, slice_idx=None):
    if axis == 0:
        data = image[slice_idx, :, :]
    elif axis == 1:
        data = image[:, slice_idx, :]
    elif axis == 2:
        data = image[:, :, slice_idx]
    
    # Flatten the data to a 1D array for statistical operations
    data_flat = jnp.abs(data.flatten())
    
    # Calculate the 25th and 75th percentiles (Q1 and Q3)
    q1 = jnp.percentile(data_flat, 40)
    q3 = jnp.percentile(data_flat, 60)
    
    # Calculate the IQR (Interquartile Range)
    iqr = q3 - q1
    
    # Define bounds for removing outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out values that are outside the IQR range
    filtered_data = data_flat[(data_flat >= lower_bound) & (data_flat <= upper_bound)]
    
    # Calculate the maximum of the filtered data
    return filtered_data.max()

def get_matching_adni_scan(path):
    image = os.path.basename(path)
    adni_dir = os.path.dirname(os.path.dirname(path))

    # Search the 1.5T and 3T directories for the given image_id
    pattern = re.compile(r'ADNI_(\d+_S_\d+)_MR_([A-Za-z0-9_]+)_br_raw_(\d+)_([0-9]+)_S[0-9]+_(I\d+)')
    match = pattern.match(image)
    
    if match:
        # Extract patient ID, description, and timestamp from the filename
        patient_id = match.group(1)
        query_description = match.group(2)
        timestamp_raw = match.group(3)
        datestamp = datetime.strptime(timestamp_raw, '%Y%m%d%H%M%S%f').date()
        datestamp = str(datestamp).replace("-", "")

        # Check if the image is in the 1.5T or 3T directory
        if query_description == "Axial_PD_T2_FSE_":
            target_strengh = "3T"
            target_description = "Double_TSE"
        elif query_description == "Double_TSE":
            target_strengh = "1.5T"
            target_description = "Axial_PD_T2_FSE_"

        # If image is in 1.5T, search in 3T and vice versa
        target_dir = os.path.join(adni_dir, target_strengh)

        # Iterate through the files names in the target directory
        target = None
        for file in os.listdir(target_dir):
            search_pattern = re.compile(rf'ADNI_{patient_id}_MR_{target_description}_br_raw_{datestamp}.*')

            if search_pattern.match(file):
                target = os.path.join(target_dir, file)
                break

        # Return the paths of the images with the given image_id in order of 1.5T and 3T
        if target_strengh == "1.5T":
            return target, path
        elif target_strengh == "3T":
            return path, target
        else:
            print(f"No matching image found in {target_strengh} directory for {image}")
            return None

# TODO: Make it so that dataset is optional, and if not provided, it will return all sequences and datasets
def get_brats_paths(data_dir, seq=None, dataset=None):
    datasets = [dataset] if dataset else ["BraSyn", "GLI", "GoAT", "LocalInpainting", "MEN-RT", "MET", "PED", "SSA"]
    train_paths, validate_paths = [], []

    for dataset in datasets:
        sequences = [seq] if seq else ["t1c", "t1n", "t2f", "t2w"]
        if dataset in ["BraSyn", "GLI", "MET", "PED"]:
            for split, paths in [("train", train_paths), ("validate", validate_paths)]:
                dir_path = os.path.join(data_dir, dataset, split)
                paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]
        elif dataset == "GoAT":
            for split, paths in [("train-WithOutGroundTruth", train_paths), ("validate", validate_paths)]:
                dir_path = os.path.join(data_dir, dataset, split)
                paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]
        elif dataset == "LocalInpainting":
            seq = "t1n"
            dir_path = os.path.join(data_dir, dataset, "train")
            train_paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path)]
        elif dataset == "MEN-RT":
            seq = "t1c"
            for split, paths in [("train", train_paths), ("validate", validate_paths)]:
                dir_path = os.path.join(data_dir, dataset, split)
                paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path)]
        elif dataset == "SSA":
            dir_path = os.path.join(data_dir, dataset, "train")
            train_paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]

    return train_paths, validate_paths

def get_adni_paths(data_dir):
    scans_dir = os.path.join(data_dir, "scans")
    t1_5 = []
    t3 = []
    for patient_dir in tqdm(os.listdir(scans_dir)):
        for scan_dir in os.listdir(os.path.join(scans_dir, patient_dir)):
            if scan_dir.endswith("Axial_PD_T2_FSE"):
                visit_dir = os.path.join(scans_dir, patient_dir, scan_dir)
                for visit in os.listdir(visit_dir):
                    for image_dir in os.listdir(os.path.join(visit_dir, visit)):
                        t1_5.append(os.path.join(visit_dir, visit, image_dir))

            elif scan_dir.endswith("Double_TSE"):
                visit_dir = os.path.join(scans_dir, patient_dir, scan_dir)
                for visit in os.listdir(visit_dir):
                    for image_dir in os.listdir(os.path.join(visit_dir, visit)):
                        t3.append(os.path.join(visit_dir, visit, image_dir))
    return t1_5, t3

def get_seg_paths(path):
    base_path = os.path.dirname(path)
    file_name = os.path.basename(path)

    split_name = file_name.split(".")
    CSF_name = split_name[0] + "_fast_pve_0.nii.gz"
    GM_name = split_name[0] + "_fast_pve_1.nii.gz"
    WM_name = split_name[0] + "_fast_pve_2.nii.gz"
    
    CSF = os.path.join(base_path, CSF_name)
    GM = os.path.join(base_path, GM_name)
    WM = os.path.join(base_path, WM_name)
    return CSF, GM, WM