import os
from tqdm import tqdm
from src.utils import read_nifti, read_metaimage, get_brats_paths, get_picai_paths, display
from src.undersampling_sim import convert_to_kspace, convert_to_image, undersampling

if __name__ == "__main__":
    dataset_type = "prostate"                      # brain, prostate
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if dataset_type == "brain":
        seq = "t2f"                                 # t1c, t1n, t2f, t2w
        dataset = "GLI"                             # BraSyn, GLI
        
        data_path = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
        paths, validate_paths = get_brats_paths(data_path, seq, dataset)

    elif dataset_type == "prostate":
        seq = "sag"                                 # adc, cor, hbv, sag, t2w
        fold = 0

        data_path = os.path.join(data_path, "data-picai")
        paths = get_picai_paths(data_path, fold, seq)

    paths_set = set(paths)

    real_images = []
    real_kspaces = []
    simulated_kspaces = []
    simulated_images = []
    
    undersampling_method = "radial"       # random, cartesian, radial, variable_density
    undersampling_factor = 80
    axis = 0                                        # 0: side view, 1: front view, 2: top view
    
    for path in tqdm(paths):
        if dataset_type == "brain":
            image = read_nifti(path)
        elif dataset_type == "prostate":
            image = read_metaimage(path)

        kspace = convert_to_kspace(image)
        simulated_kspace = undersampling(kspace, method=undersampling_method, axis=axis, factor=undersampling_factor)
        simulated_image = convert_to_image(simulated_kspace)

        if path in paths_set:
            real_images.append(image)
            real_kspaces.append(kspace)
            simulated_kspaces.append(simulated_kspace)
            simulated_images.append(simulated_image)

    display(real_images[0], real_kspaces[0], simulated_kspaces[0], simulated_images[0], axis=axis, highlight=True)