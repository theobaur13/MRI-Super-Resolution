import os
import argparse
import matplotlib.pyplot as plt
from src.readwrite import *
from src.paths import *
from src.analysis import *
from src.adni import *
from src.sampling import *
from src.utils import *
from src.kspace import *
from src.display import *
from src.gibbs_removal import *

# Axis 0: Axial, Axis 1: Sagittal, Axis 2: Coronal

def generate_simulated_image(kspace, axis):
    simulated_kspace = radial_undersampling(kspace, axis=axis, factor=0.7)
    simulated_kspace = gaussian_plane(simulated_kspace, axis=0, sigma=0.5, mu=0.5, A=2)
    simulated_image = convert_to_image(simulated_kspace)

    # simulated_image = jax_to_numpy(simulated_image)
    # simulated_image = gibbs_removal(simulated_image, slice_axis=axis)
    # simulated_image = numpy_to_jax(simulated_image)
    
    simulated_image = gaussian_plane(simulated_image, axis=0, sigma=0.4, mu=0.5, A=1, invert=True)
    simulated_image = random_noise(simulated_image, intensity=0.01, frequency=0.3)
    return simulated_image, simulated_kspace

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    ADNI_dir = os.path.join(data_path, "ADNI")
    ADNI_collapsed_dir = os.path.join(data_path, "ADNI_collapsed")
    brats_dir = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
    brats_output_dir = os.path.join(data_path, "BraTS_output")

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["organise-adni", "simulate", "analyse", "batch-convert", "view"], help="Action to perform")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
    parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")

    # Analysis arguments
    parser.add_argument("--dataset", type=str, help="Dataset for analysis (e.g., 'ADNI', 'BraTS')")
    parser.add_argument("--subject", type=str, help="Subject for analysis (e.g., 'noise', 'brightness')")
    parser.add_argument("--index", type=int, help="Index of ADNI image to simulate")

    # Batch conversion arguments
    parser.add_argument("--seq", type=str, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    parser.add_argument("--brats_dataset", type=str, help="Dataset type (e.g., 'BraSyn', 'GLI')")

    # View arguments
    parser.add_argument("--path", type=str, help="Path to NIfTI file to display")
    parser.add_argument("--type", type=str, choices=["NIfTI", "DICOM"], help="Type of file to display")

    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    # > py main.py organise-adni
    if action == "organise-adni":
        collapse_adni(ADNI_dir, ADNI_collapsed_dir)

    # Apply degradation to ADNI scans
    # > py main.py simulate --index 0 --axis 0 --slice 24
    elif action == "simulate":
        # Arguments
        index = args.index
        axis = args.axis
        slice_idx = args.slice

        df = adni_dataframe(ADNI_collapsed_dir)
        paths_1_5T, paths_3T = get_adni_pair(df, index)
        
        image_1_5T = read_dicom(paths_1_5T)
        image_3T = read_dicom(paths_3T, flip=True)
        image_3T = image_3T[0:48, 0:256, 0:256]

        kspace_1_5T = convert_to_kspace(image_1_5T)
        kspace_3T = convert_to_kspace(image_3T)
        simulated_image, simulated_kspace = generate_simulated_image(kspace_3T, axis=0)

        max_value = max(
            robust_max(kspace_1_5T, axis, slice_idx),
            robust_max(kspace_3T, axis, slice_idx),
            robust_max(simulated_kspace, axis, slice_idx)
        ) * slice_idx * 10

        display_comparison(image_1_5T, image_3T, slice=slice_idx, axis=axis, kspace=False)
        display_comparison(image_1_5T, simulated_image, slice=slice_idx, axis=axis, kspace=False)
        plot_3d_surfaces([kspace_1_5T, kspace_3T, simulated_kspace], slice_idx, axis=axis, cmap="viridis", limit=max_value)
        plt.show()

    # Analyse central brightness of ADNI scans
    elif action == "analyse":
        # Arguments
        subject = args.subject
        axis = args.axis
        limit = args.limit
        datset = args.dataset.lower()

        if datset == "adni":
            df = adni_dataframe(ADNI_collapsed_dir)

            # ADNI has flatter scans than BraTS
            hypervolume_1_5T = np.zeros((limit, 48, 256, 256))
            hypervolume_3T = np.zeros((limit, 48, 256, 256))
        elif datset == "brats":
            train_paths, validate_paths = get_brats_paths(brats_dir, "t2f", "BraSyn")
            paths_3T = train_paths + validate_paths
            paths_1_5T = train_paths + validate_paths   # Placeholder for 1.5T paths

            hypervolume_1_5T = np.zeros((limit, 155, 240, 240))
            hypervolume_3T = np.zeros((limit, 155, 240, 240))

        for i in tqdm(range(limit)):
            if datset == "adni":
                paths_1_5T, paths_3T = get_adni_pair(df, i)
                image_1_5T = read_dicom(paths_1_5T)
                image_3T = read_dicom(paths_3T, flip=True)

                image_1_5T = image_1_5T[0:48, 0:256, 0:256]
                image_3T = image_3T[0:48, 0:256, 0:256]
            elif datset == "brats":
                image_1_5T = read_nifti(paths_1_5T[i], brats=True)
                image_3T = read_nifti(paths_3T[i], brats=True)

            hypervolume_1_5T[i] = image_1_5T
            hypervolume_3T[i] = image_3T

        # > py main.py analyse --subject "noise" --dataset "ADNI" --axis 0 --limit 52
        # > py main.py analyse --subject "noise" --dataset "BraTS" --axis 0 --limit 15
        if subject == "noise":
            compare_snr(hypervolume_1_5T, hypervolume_3T, axis)

        # > py main.py analyse --subject "brightness" --dataset "ADNI" --slice 24 --axis 0 --limit 52
        # > py main.py analyse --subject "brightness" --dataset "BraTS" --slice 65 --axis 0 --limit 15
        elif subject == "brightness":
            slice_idx = args.slice
            generate_brightness_mask(hypervolume_1_5T, hypervolume_3T, slice_idx, axis=axis, sigma=20, lim=0.6)

        # > py main.py analyse --subject "gibbs" --dataset "ADNI" --index 0 --slice 24 --axis 0 --limit 52
        # > py main.py analyse --subject "gibbs" --dataset "BraTS" --index 0 --slice 24 --axis 0 --limit 15
        elif subject == "gibbs":
            # Arguments
            index = args.index
            slice_idx = args.slice

            original_volume = hypervolume_3T[index, :, :, :]
            kspace = convert_to_kspace(original_volume)
            kspace = radial_undersampling(kspace, axis=axis, factor=0.7)
            kspace = gaussian_plane(kspace, axis=0, sigma=0.5, mu=0.5, A=2)
            simulated_volume = convert_to_image(kspace)
            reduced_volume = gibbs_removal(jax_to_numpy(simulated_volume), slice_axis=axis)

            display_comparison(original_volume, reduced_volume, slice=slice_idx, axis=axis, kspace=False)
            display_comparison(simulated_volume, reduced_volume, slice=slice_idx, axis=axis, kspace=False)

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
            image = read_nifti(path)
            kspace = convert_to_kspace(image)
            simulated_image, simulated_kspace = generate_simulated_image(kspace, axis=axis)

            write_nifti(simulated_image, os.path.join(brats_output_dir, os.path.basename(path)))

    # > py main.py view --path "BraTS_output/BraTS-GLI-00000-000-t2f.nii.gz" --type "NIfTI" --slice 65 --axis 0
    # > py main.py view --path "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000/BraSyn/train/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz" --type "NIfTI" --slice 65 --axis 0
    # > py main.py view --path "ADNI_collapsed/ADNI_002_S_0413_MR_Axial_PD_T2_FSE__br_raw_20061115094759_1_S22556_I29704.dcm" --type "DICOM" --slice 24 --axis 0
    elif action == "view":
        # Arguments
        relative_path = args.path
        scan_type = args.type.lower()

        if scan_type == "dicom":
            df = adni_dataframe(ADNI_collapsed_dir)
            image_id = relative_path.split("/")[-1].split("_")[-1].split(".")[0]
            paths = adni_search_by_id(df, image_id)
            image = read_dicom(paths, flip=True)
            
        elif scan_type == "nifti":
            absolute_path = os.path.join(data_path, relative_path)
            image = read_nifti(absolute_path, brats=True)

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        plot_slice(ax, image, slice=args.slice, axis=args.axis)
        plt.show()