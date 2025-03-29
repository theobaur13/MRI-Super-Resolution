import os
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from src.utils import read_nifti, read_metaimage, get_brats_paths, get_picai_paths, display, convert_to_tensor
from src.undersampling_sim import convert_to_kspace, convert_to_image, random_undersampling, cartesian_undersampling, radial_undersampling, variable_density_undersampling, downsize_kspace
from src.training import SRCNN_MRI, training_loop

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

        data_path = os.path.join(data_path, "data-picai-main")
        paths = get_picai_paths(data_path, fold, seq)

    real_images = []
    real_kspaces = []
    simulated_kspaces = []
    simulated_images = []
    
    axis = 0                                                # 0: side view, 1: front view, 2: top view
    
    for path in tqdm(paths):
        if dataset_type == "brain":
            image = read_nifti(path)
        elif dataset_type == "prostate":
            image = read_metaimage(path)
        
        kspace = convert_to_kspace(image)
        simulated_kspace = radial_undersampling(kspace, axis=axis, factor=0.5)
        simulated_kspace = random_undersampling(simulated_kspace, factor=1.05)
        simulated_kspace = downsize_kspace(simulated_kspace, size=256)
        simulated_image = convert_to_image(simulated_kspace)
        
        real_images.append(image)
        real_kspaces.append(kspace)
        simulated_kspaces.append(simulated_kspace)
        simulated_images.append(simulated_image)

    index = 0
    display(real_images[index],
            real_kspaces[index],
            simulated_kspaces[index],
            simulated_images[index],
            axis=axis,
            highlight=True
    )

    model = SRCNN_MRI(num_channels=2) # 2 channels for real and imaginary parts
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 10
    batch_size = 1
    
    LR = convert_to_tensor(simulated_images, slice_axis=axis)
    HR = convert_to_tensor(real_images, slice_axis=axis)

    print("LR shape:", LR.shape)  # Expected: (num_slices, 2, H, W)
    print("HR shape:", HR.shape)  # Expected: (num_slices, 2, H, W)

    training_loop(model, optimizer, criterion, epochs, LR, HR)