import os

def get_brats_paths(data_dir, seq=None, dataset=None):
    datasets = [dataset] if dataset else ["BraSyn", "GLI", "GoAT", "MET", "SSA"]
    train_paths, validate_paths = [], []

    for dataset in datasets:
        sequences = [seq] if seq else ["t2f"]
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