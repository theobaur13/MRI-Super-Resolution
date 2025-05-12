import os
import argparse
import nibabel as nib
from src.readwrite import *
from src.paths import *
from src.analysis import *
from src.adni import *
from src.sampling import *
from src.utils import *
from src.kspace import *
from src.display import *
from src.gibbs_removal import *

# Axis 0: Saggital, Axis 1: Coronal, Axis 2: Axial

# Degrade 3T scans to resemble 1.5T scans
def simluation_pipeline(nifti, axis, visualize=False, slice=None):
    original_volume = jnp.array(nifti.get_fdata())
    original_kspace = convert_to_kspace(original_volume)

    # k-space manipulation
    cylindrical_cropped_kspace = cylindrical_crop(original_kspace, axis=axis, factor=0.7)
    gaussian_amped_kspace = gaussian_amplification(cylindrical_cropped_kspace, axis=0, sigma=0.5, mu=0.5, A=2)
    
    # Image manipulation
    simulated_volume = convert_to_image(gaussian_amped_kspace)
    gibbs_reduced_volume = numpy_to_jax(gibbs_removal(jax_to_numpy(simulated_volume), slice_axis=axis))
    gaussian_amped_image = gaussian_amplification(gibbs_reduced_volume, axis=0, sigma=0.4, mu=0.5, A=1, invert=True)
    noisy_image = random_noise(gaussian_amped_image, intensity=0.01, frequency=0.3)
    
    if visualize:
        if slice is None:
            raise ValueError("Slice index must be provided for visualization.")
        
        # Calculate the correct slice index for the given axis
        real_world_slice = slice
        voxel_slice = world_to_voxel_slice(real_world_slice, axis, nifti.affine)

        # Display the original and simulated images
        display_comparison_volumes([
            original_volume, simulated_volume, gibbs_reduced_volume, gaussian_amped_image, noisy_image], 
            slice=voxel_slice, axis=axis)

    # Convert to NIfTI
    simulated_nifti = nib.Nifti1Image(jax_to_numpy(noisy_image), affine=nifti.affine)
    return simulated_nifti, gaussian_amped_kspace

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    ADNI_dir = os.path.join(data_path, "ADNI")
    ADNI_nifti_dir = os.path.join(data_path, "ADNI_NIfTIs")
    brats_dir = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
    brats_output_dir = os.path.join(data_path, "BraTS_output")

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[
        "convert-adni",
        "simulate",
        "analyse-noise",
        "analyse-brightness",
        "batch-convert",
        "view"
        ], help="Action to perform")
    parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")

    # View and simulate arguments
    parser.add_argument("--path", type=str, help="Relative path to NIfTI file to display")

    # Analysis arguments
    parser.add_argument("--dataset", type=str, help="Dataset for analysis (e.g., 'ADNI', 'BraTS')")

    # Batch conversion arguments
    parser.add_argument("--seq", type=str, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    parser.add_argument("--brats_dataset", type=str, help="Dataset type (e.g., 'BraSyn', 'GLI')")

    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    # > py main.py convert-adni
    if action == "convert-adni":
        convert_adni(ADNI_dir, ADNI_nifti_dir)

    # Apply degradation to slice in a volume
    # > py main.py simulate --path "data/ADNI_NIfTIs/3T/ADNI_002_S_0413_MR_Double_TSE_br_raw_20061115141733_1_S22682_I30117.nii.gz" --axis 2 --slice 24
    elif action == "simulate":
        # Arguments
        axis = args.axis
        slice_idx = args.slice
        arg_path = args.path
        path = os.path.join(base_dir, arg_path)

        path_1_5T, path_3T = get_matching_adni_scan(path)
        
        nifti_1_5T = read_nifti(path_1_5T)
        nifti_3T = read_nifti(path_3T)

        simulated_nifti, simulated_kspace = simluation_pipeline(nifti_3T, axis=axis, visualize=True, slice=slice_idx)

        display_comparison_niftis([nifti_1_5T, nifti_3T], slice=slice_idx, axis=axis)
        display_comparison_niftis([nifti_1_5T, simulated_nifti], slice=slice_idx, axis=axis)
        plt.show()

    # Perform analysis between two types of scans
    # > py main.py analyse-noise --dataset "ADNI" --axis 0 
    # > py main.py analyse-noise --dataset "BraTS" --axis 0 
    # > py main.py analyse-brightness --dataset "ADNI" --slice 24 --axis 0
    # > py main.py analyse-brightness --dataset "BraTS" --slice 65 --axis 0
    elif action == "analyse-noise" or action == "analyse-brightness":
        # Arguments
        axis = args.axis
        datset = args.dataset.lower()

        # Sort out paths for ADNI and BraTS
        if datset == "adni":
            shape = (256, 256, 44)
            paths_1_5T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "1.5T")))
            paths_3T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "3T")))
            paths_1_5T = [os.path.join(ADNI_nifti_dir, "1.5T", p) for p in paths_1_5T]
            paths_3T = [os.path.join(ADNI_nifti_dir, "3T", p) for p in paths_3T]

        elif datset == "brats":
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

    # Apply degradation to BraTS scans
    # > py main.py batch-convert --seq "t2f" --brats_dataset "BraSyn"
    elif action == "batch-convert":
        # Arguments
        seq = args.seq                  # t1c, t1n, t2f, t2w
        brats_dataset = args.brats_dataset          # BraSyn, GLI
    
        os.makedirs(brats_output_dir, exist_ok=True)

        paths, validate_paths = get_brats_paths(brats_dir, seq, brats_dataset)

        axis = 0
        for path in tqdm(paths):
            nifti = read_nifti(path)
            simulated_nifti, _ = simluation_pipeline(nifti, axis=axis)
            write_nifti(simulated_nifti, os.path.join(brats_output_dir, os.path.basename(path)))

    # > py main.py view --path "data/BraTS_output/BraTS-GLI-00000-000-t2f.nii.gz" --slice 65 --axis 2
    # > py main.py view --path "data/data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000/BraSyn/train/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz" --slice 65 --axis 2
    # > py main.py view --path "data/ADNI_NifTIs/1.5T/ADNI_002_S_0413_MR_Axial_PD_T2_FSE__br_raw_20061115094759_1_S22556_I29704.nii.gz" --slice 24 --axis 2
    elif action == "view":
        # Arguments
        relative_path = args.path
        absolute_path = os.path.join(base_dir, relative_path)

        nifti = read_nifti(absolute_path)

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        plot_slice_from_nifti(ax, nifti, slice=args.slice, axis=args.axis)
        plt.show()