import os
from tqdm import tqdm
from src.utils import read_nifti, plot_matrix, get_paths
from src.undersampling_sim import convert_to_kspace, convert_to_image, undersampling

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
    seq = "t1c"
    dataset = "BraSyn"

    train_paths, validate_paths = get_paths(data_path, seq, dataset)
    train_paths_set = set(train_paths)

    train_images = []
    train_kspace = []
    train_simulated_kspace = []
    train_simulated_images = []
    validate_images = []
    validate_kspace = []
    validate_simulated_kspace = []
    validate_simulated_images = []
    
    undersampling_method = "radial"
    undersampling_factor = 4
    axis = 0
    
    for path in tqdm(train_paths + validate_paths):
        image = read_nifti(path)
        kspace = convert_to_kspace(image)
        simulated_kspace = undersampling(kspace, method=undersampling_method, axis=axis, factor=undersampling_factor)
        simulated_image = convert_to_image(simulated_kspace)

        if path in train_paths_set:
            train_images.append(image)
            train_kspace.append(kspace)
            train_simulated_kspace.append(simulated_kspace)
            train_simulated_images.append(simulated_image)
        else:
            validate_images.append(image)
            validate_kspace.append(kspace)
            validate_simulated_kspace.append(simulated_kspace)
            validate_simulated_images.append(simulated_image)


    plot_matrix(train_images[0], axis=axis)
    plot_matrix(train_kspace[0], axis=axis)
    plot_matrix(train_simulated_kspace[0], axis=axis)
    plot_matrix(train_simulated_images[0], axis=axis)