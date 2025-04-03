import os
from tqdm import tqdm

def get_brats_paths(data_dir, seq, dataset):
    train_dir = os.path.join(data_dir, dataset, "train")
    validate_dir = os.path.join(data_dir, dataset, "validate")

    train_paths = [os.path.join(train_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(train_dir)]
    validate_paths = [os.path.join(validate_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(validate_dir)]

    return train_paths, validate_paths

def get_picai_paths(data_dir, fold, seq, limit=1):
    dir = os.path.join(data_dir, "images", f"fold{fold}")

    paths = []
    for patient in tqdm(os.listdir(dir)):
        patient_dir = os.path.join(dir, patient)
        for file in os.listdir(patient_dir):
            if file.endswith(f"{seq}.mha"):
                paths.append(os.path.join(patient_dir, file))
                if len(paths) == limit:
                    return paths
    return paths

def get_ixi_paths(data_dir, limit=2):
    t1_5 = []
    t3 = []
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".nii.gz") and "HH" in file:
            t3.append(os.path.join(data_dir, file))
        elif file.endswith(".nii.gz") and "HH" not in file:
            t1_5.append(os.path.join(data_dir, file))
    t1_5 = t1_5[:limit]
    t3 = t3[:limit]

    return t1_5, t3