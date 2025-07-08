import os

def get_brats_paths(data_dir, seq=None, dataset=None):
    datasets = [dataset] if dataset else ["BraSyn", "GLI", "GoAT", "MET", "SSA"]
    train_paths, validate_paths = [], []

    for dataset in datasets:
        sequences = [seq] if seq else ["t2f", "seg"]
        if dataset in ["BraSyn", "GLI", "MET"]:
            for split, paths in [("train", train_paths), ("validate", validate_paths)]:
                dir_path = os.path.join(data_dir, dataset, split)
                paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]
        elif dataset == "GoAT":
            for split, paths in [("train-WithOutGroundTruth", train_paths), ("validate", validate_paths)]:
                dir_path = os.path.join(data_dir, dataset, split)
                paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]
        elif dataset == "SSA":
            dir_path = os.path.join(data_dir, dataset, "train")
            train_paths += [os.path.join(dir_path, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(dir_path) for seq in sequences]

    return train_paths, validate_paths

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