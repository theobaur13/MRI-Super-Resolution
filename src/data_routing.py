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
                        print(os.path.join(visit_dir, visit, image_dir))
                        t3.append(os.path.join(visit_dir, visit, image_dir))

    t1_5 = t1_5[:limit]
    t3 = t3[:limit]
    return t1_5, t3
                
    