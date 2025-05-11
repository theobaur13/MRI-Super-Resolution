import os
from tqdm import tqdm

def get_brats_paths(data_dir, seq, dataset=None):
    datasets = [dataset] if dataset else ["BraSyn", "GLI"]
    train_paths, validate_paths = [], []

    for dset in datasets:
        for split, paths in [("train", train_paths), ("validate", validate_paths)]:
            dir_path = os.path.join(data_dir, dset, split)
            paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path)]

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