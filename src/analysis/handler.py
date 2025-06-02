import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.slicing import slice_nifti
from src.utils.paths import get_brats_paths
from src.utils.readwrite import read_nifti
from src.analysis.analysis import (
    compare_snr,
    generate_brightness_map,
    generate_snr_map
)

def analyse(args):
    # Arguments
    action = args.action.lower()
    axis = args.axis
    dataset_dir = args.dataset_dir

    # Sort out paths for ADNI and BraTS
    if "adni" in dataset_dir.lower():
        shape = (256, 256, 44)
        paths_1_5T = sorted(os.listdir(os.path.join(dataset_dir, "1.5T")))
        paths_3T = sorted(os.listdir(os.path.join(dataset_dir, "3T")))
        paths_1_5T = [os.path.join(dataset_dir, "1.5T", p) for p in paths_1_5T]
        paths_3T = [os.path.join(dataset_dir, "3T", p) for p in paths_3T]

    elif "brats" in dataset_dir.lower():
        dataset = args.dataset
        seq = args.seq
        shape = (240, 240, 155)
        train_paths, validate_paths = get_brats_paths(dataset_dir, seq, dataset)
        paths_1_5T = train_paths + validate_paths  # Placeholder pairing
        paths_3T = train_paths + validate_paths

    # Load nifti files (remember to implement limit at some point)
    niftis_1_5T = [read_nifti(path) for path in tqdm(paths_1_5T)]
    niftis_3T = [read_nifti(path) for path in tqdm(paths_3T)]

    # Compare SNR at each slice between 1.5T and 3T scans
    if action == "analyse-snr-avg":
        # Allocate hypervolumes
        hypervolume_1_5T = np.zeros((len(niftis_1_5T), *shape))
        for i, nifti in tqdm(enumerate(niftis_1_5T)):
            hypervolume_1_5T[i] = jnp.array(nifti.get_fdata())[0:shape[0], 0:shape[1], 0:shape[2]]
        
        hypervolume_3T = np.zeros((len(niftis_3T), *shape))
        for i, nifti in tqdm(enumerate(niftis_3T)):
            hypervolume_3T[i] = jnp.array(nifti.get_fdata())[0:shape[0], 0:shape[1], 0:shape[2]]

        compare_snr(hypervolume_1_5T, hypervolume_3T, axis)

    # Compare brightness at a certain point on certain axis between 1.5T and 3T scans
    elif action == "analyse-brightness" or action == "analyse-snr-map":
        slice_idx = args.slice
        
        slices_1_5T = []
        for nifti in tqdm(niftis_1_5T):
            slice = slice_nifti(nifti, slice_idx, axis)
            slices_1_5T.append(slice)

        slices_3T = []
        for nifti in tqdm(niftis_3T):
            slice = slice_nifti(nifti, slice_idx, axis)
            slices_3T.append(slice)

        slices_1_5T = jnp.array(slices_1_5T)
        slices_3T = jnp.array(slices_3T)

        if action == "analyse-brightness":
            generate_brightness_map(slices_3T, slices_1_5T)
            
        elif action == "analyse-snr-map":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

            map_1_5T = generate_snr_map(slices_1_5T)
            map_3T = generate_snr_map(slices_3T)

            # Display the SNR maps
            im1 = ax1.imshow(map_1_5T, cmap="plasma")
            im2 = ax2.imshow(map_3T, cmap="plasma")

            minimum = min(np.min(map_1_5T), np.min(map_3T))
            maximum = max(np.max(map_1_5T), np.max(map_3T))

            im1.set_clim(minimum, maximum)
            im2.set_clim(minimum, maximum)

            ax1.set_title("SNR Map 1.5T")
            ax2.set_title("SNR Map 3T")
            fig.colorbar(im1, ax=ax1)
            fig.colorbar(im2, ax=ax2)
            plt.tight_layout()

    plt.show()