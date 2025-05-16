import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import dicom2nifti
from src.pipeline import simluation_pipeline
from src.conversions import convert_to_kspace
from src.readwrite import read_nifti, write_nifti
from src.display import display_img, display_3d
from src.analysis import compare_snr, generate_brightness_mask, generate_snr_map
from src.slicing import slice_nifti
from src.utils import get_adni_paths, get_matching_adni_scan, get_brats_paths

def convert_adni(args, base_dir):
    ADNI_dir = os.path.join(base_dir, args.ADNI_dir)
    output_dir = os.path.join(base_dir, args.ADNI_nifti_dir)

    t1_5_paths, t3_paths = get_adni_paths(ADNI_dir)

    # Create 1.5T and 3T subdirectories
    os.makedirs(os.path.join(output_dir, "1.5T"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "3T"), exist_ok=True)

    def process(paths, target_dir):
        for dir in tqdm(paths):
            # Edit timestamps to reflect the directory name
            timestamp = dir.split("\\")[-2]
            timestamp = timestamp.replace(".0", "")
            timestamp = timestamp.replace("_", "")
            timestamp = timestamp.replace("-", "")
            contents = os.listdir(dir)
            
            old_name = contents[0]
            parts = old_name.split("_")
            parts[-4] = timestamp
            new_name = "_".join(parts)
            new_name = new_name.replace(".dcm", ".nii.gz")

            # Convert the volume to a NIfTI image
            dicom2nifti.dicom_series_to_nifti(dir, os.path.join(target_dir, new_name), reorient_nifti=True)

    process(t1_5_paths, os.path.join(output_dir, "1.5T"))
    process(t3_paths, os.path.join(output_dir, "3T"))

def simulate(args, base_dir):
    # Arguments
    axis = args.axis
    slice_idx = args.slice
    arg_path = args.path
    path = os.path.join(base_dir, arg_path)

    path_1_5T, path_3T = get_matching_adni_scan(path)
    
    nifti_1_5T = read_nifti(path_1_5T)
    nifti_3T = read_nifti(path_3T)

    simulated_nifti, simulated_kspace = simluation_pipeline(nifti_3T, axis=axis, visualize=True, slice=slice_idx)

    # Display the target vs simulated image
    display_img([nifti_1_5T, simulated_nifti], slice=slice_idx, axis=axis, titles=["Original 1.5T Image", "Simulated 1.5T Image"])

    # Display the target vs simulated k-space
    original_volume = jnp.array(nifti_1_5T.get_fdata())
    original_kspace = convert_to_kspace(original_volume)
    display_3d([original_kspace, simulated_kspace], slice=slice_idx, axis=axis, limit=1, titles=["Target 1.5T k-Space", "Simulated 1.5T k-Space"])
    plt.show()

def analyse(args, base_dir):
    # Arguments
    action = args.action.lower()
    axis = args.axis
    datset = args.dataset.lower()

    # Sort out paths for ADNI and BraTS
    if datset == "adni":
        ADNI_nifti_dir = os.path.join(base_dir, "data", "ADNI_NIfTIs")
        shape = (256, 256, 44)
        paths_1_5T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "1.5T")))
        paths_3T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "3T")))
        paths_1_5T = [os.path.join(ADNI_nifti_dir, "1.5T", p) for p in paths_1_5T]
        paths_3T = [os.path.join(ADNI_nifti_dir, "3T", p) for p in paths_3T]

    elif datset == "brats":
        brats_dir = os.path.join(base_dir, "data", "BraTS_NifTIs")
        shape = (240, 240, 155)
        train_paths, validate_paths = get_brats_paths(brats_dir, "t2f", "BraSyn")
        paths_1_5T = train_paths + validate_paths  # Placeholder pairing
        paths_3T = train_paths + validate_paths

    # Load nifti files (remember to implement limit at some point)
    niftis_1_5T = [read_nifti(path) for path in tqdm(paths_1_5T)]
    niftis_3T = [read_nifti(path) for path in tqdm(paths_3T)]

    # Compare SNR at each slice between 1.5T and 3T scans
    if action == "analyse-noise":
        # Allocate hypervolumes
        hypervolume_1_5T = np.zeros((len(niftis_1_5T), *shape))
        for i, nifti in enumerate(niftis_1_5T):
            hypervolume_1_5T[i] = jnp.array(nifti.get_fdata())[0:shape[0], 0:shape[1], 0:shape[2]]
        
        hypervolume_3T = np.zeros((len(niftis_3T), *shape))
        for i, nifti in enumerate(niftis_3T):
            hypervolume_3T[i] = jnp.array(nifti.get_fdata())[0:shape[0], 0:shape[1], 0:shape[2]]

        compare_snr(hypervolume_1_5T, hypervolume_3T, axis)

        # TODO: Create SNR map for given slice.
        generate_snr_map()

    # Compare brightness at a certain point on certain axis between 1.5T and 3T scans
    elif action == "analyse-brightness":
        slice_idx = args.slice
        
        slices_1_5T = []
        for nifti in niftis_1_5T:
            slice = slice_nifti(nifti, slice_idx, axis)
            slices_1_5T.append(slice)

        slices_3T = []
        for nifti in niftis_3T:
            slice = slice_nifti(nifti, slice_idx, axis)
            slices_3T.append(slice)

        slices_1_5T = jnp.array(slices_1_5T)
        slices_3T = jnp.array(slices_3T)
        generate_brightness_mask(slices_1_5T, slices_3T, axis=axis, sigma=20, lim=0.6)

    plt.show()

def batch_convert(args, base_dir):
    # Arguments
    seq = args.seq                                          # t1c, t1n, t2f, t2w
    dataset = args.brats_dataset                            # BraSyn, GLI
    output_dir = os.path.join(base_dir, args.output_dir)    # Output directory for converted data
    brats_dir = os.path.join(base_dir, "data", "BraTS_NifTIs", dataset)  # Path to BraTS dataset

    os.makedirs(output_dir, exist_ok=True)
    paths, validate_paths = get_brats_paths(brats_dir, seq, dataset)

    axis = 0
    for path in tqdm(paths):
        nifti = read_nifti(path)
        simulated_nifti, _ = simluation_pipeline(nifti, axis=axis)
        write_nifti(simulated_nifti, os.path.join(output_dir, os.path.basename(path)))

def view(args, base_dir):
    relative_path = args.path
    absolute_path = os.path.join(base_dir, relative_path)

    nifti = read_nifti(absolute_path)
    display_img([nifti], slice=args.slice, axis=args.axis)
    plt.show()