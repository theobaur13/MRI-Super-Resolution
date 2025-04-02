import os
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from src.utils import read_nifti, read_metaimage, get_brats_paths, get_picai_paths, display, convert_to_tensor
from src.simulation import convert_to_kspace, convert_to_image, random_undersampling, cartesian_undersampling, radial_undersampling, variable_density_undersampling, downsize_kspace

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    # Command line arguments for dataset type
    if len(sys.argv) < 2:
        print("Usage: python main.py [dataset_type]")
        print("dataset_type: brain or prostate")
        sys.exit(1)
    dataset_type = sys.argv[1].lower()

    print("Collecting paths...")
    if dataset_type == "brain":
        seq = "t2f"                                 # t1c, t1n, t2f, t2w
        dataset = "BraSyn"                             # BraSyn, GLI
        
        data_path = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
        paths, validate_paths = get_brats_paths(data_path, seq, dataset)

    elif dataset_type == "prostate":
        seq = "sag"                                 # adc, cor, hbv, sag, t2w
        fold = 0

        data_path = os.path.join(data_path, "data-picai-main")
        paths = get_picai_paths(data_path, fold, seq)

    real_images = []
    real_kspaces = []
    simulated_kspaces = []
    simulated_images = []
    
    axis = 0                                                # 0: side view, 1: front view, 2: top view
    
    print("Manipulating image k-spaces...")
    for path in tqdm(paths):
        if dataset_type == "brain":
            image = read_nifti(path)
        elif dataset_type == "prostate":
            image = read_metaimage(path)
        
        kspace = convert_to_kspace(image)
        simulated_kspace = radial_undersampling(kspace, axis=axis, factor=0.4)
        simulated_kspace = random_undersampling(simulated_kspace, factor=1.05)
        simulated_kspace = downsize_kspace(simulated_kspace, axis=axis, size=128)
        simulated_image = convert_to_image(simulated_kspace)
        
        real_images.append(image)
        real_kspaces.append(kspace)
        simulated_kspaces.append(simulated_kspace)
        simulated_images.append(simulated_image)

    index = 9
    display(real_images[index],
            real_kspaces[index],
            simulated_kspaces[index],
            simulated_images[index],
            axis=axis,
            highlight=True
    )