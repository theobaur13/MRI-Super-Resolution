import os
from tqdm import tqdm
from src.utils import read_nifti, plot_matrix, get_paths
from src.undersampling_sim import convert_to_kspace

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
    seq = "t1c"
    dataset = "GLI"

    train_paths, validate_paths = get_paths(data_path, seq, dataset)
    train_paths_set = set(train_paths)

    train_images = []
    train_kspace = []
    validate_images = []
    validate_kspace = []
    
    for path in tqdm(train_paths + validate_paths):
        image = read_nifti(path)
        kspace = convert_to_kspace(image)
        if path in train_paths_set:
            train_images.append(image)
            train_kspace.append(kspace)
        else:
            validate_images.append(image)
            validate_kspace.append(kspace)

    plot_matrix(image[:, :, 75])
    plot_matrix(kspace[:, :, 75])