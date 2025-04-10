import os
from tqdm import tqdm
import shutil

def get_brats_paths(data_dir, seq, dataset):
    train_dir = os.path.join(data_dir, dataset, "train")
    validate_dir = os.path.join(data_dir, dataset, "validate")

    train_paths = [os.path.join(train_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(train_dir)]
    validate_paths = [os.path.join(validate_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(validate_dir)]

    return train_paths, validate_paths

def get_adni_paths(data_dir, limit=10):
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

    t1_5 = t1_5[:limit]
    t3 = t3[:limit]
    return t1_5, t3

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