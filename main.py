import os
import argparse
from src.readwrite import *
from src.paths import *
from src.analysis import *
from src.adni import *
from src.sampling import *
from src.utils import *
from src.kspace import *
from src.display import *

def generate_simulated_image(kspace, axis):
    simulated_kspace = radial_undersampling(kspace, axis=axis, factor=0.7)
    simulated_kspace = gaussian_plane(simulated_kspace, axis=0, sigma=0.5, mu=0.5, A=2)
    simulated_image = convert_to_image(simulated_kspace)
    simulated_image = gaussian_plane(simulated_image, axis=0, sigma=0.4, mu=0.5, A=1, invert=True)
    simulated_image = random_noise(simulated_image, intensity=0.01, frequency=0.3)
    return simulated_image, simulated_kspace

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    ADNI_dir = os.path.join(data_path, "ADNI")
    ADNI_collapsed_dir = os.path.join(data_path, "ADNI_collapsed")

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["organise-adni", "simulate", "analyse", "batch-convert"], help="Action to perform")
    parser.add_argument("--index", type=int, help="Index of ADNI image to simulate")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
    parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    # > py main.py organise-adni
    if action == "organise-adni":
        collapse_adni(ADNI_dir, ADNI_collapsed_dir)

    # Apply degradation to ADNI scans
    # > py main.py simulate --index 0
    elif action == "simulate":
        index = args.index
        df = adni_dataframe(ADNI_collapsed_dir)
        paths_1_5T, paths_3T = get_adni_pair(df, index)
        
        image_T1_5 = read_dicom(paths_1_5T)

        image_T3 = read_dicom(paths_3T, flip=True)
        image_T3 = image_T3[0:48, 0:256, 0:256]

        T1_5_kspace = convert_to_kspace(image_T1_5)
        T3_kspace = convert_to_kspace(image_T3)
        simulated_image, simulated_kspace = generate_simulated_image(T3_kspace, axis=0)

        axis = 0
        slice_idx = 24

        max_value = max(
            robust_max(T1_5_kspace, axis, slice_idx),
            robust_max(T3_kspace, axis, slice_idx),
            robust_max(simulated_kspace, axis, slice_idx)
        ) * slice_idx * 10

        display_comparison(image_T1_5, image_T3, slice=slice_idx, axis=axis, kspace=False)
        display_comparison(image_T1_5, simulated_image, slice=slice_idx, axis=axis, kspace=False)
        plot_3d_kspace([T1_5_kspace, T3_kspace, simulated_kspace], slice_idx, axis=axis, cmap="viridis", limit=max_value)
        plt.show()

    # Analyse central brightness of ADNI scans
    # > py main.py analyse --slice 24 --axis 0 --limit 5
    elif action == "analyse":
        df = adni_dataframe(ADNI_collapsed_dir)
        slice_idx = args.slice
        axis = args.axis
        limit = args.limit

        # Get the paths for the 1.5T and 3T images
        scans_1_5T = np.zeros((limit, 48, 256, 256))
        scans_3T = np.zeros((limit, 48, 256, 256))
        for i in tqdm(range(limit)):
            paths_1_5T, paths_3T = get_adni_pair(df, i)
            image_1_5T = read_dicom(paths_1_5T)
            image_1_5T = image_1_5T[0:48, 0:256, 0:256]

            image_3T = read_dicom(paths_3T, flip=True)
            image_3T = image_3T[0:48, 0:256, 0:256]

            scans_1_5T[i] = image_1_5T
            scans_3T[i] = image_3T

        generate_brightness_mask(scans_1_5T, scans_3T, slice_idx, axis=axis, sigma=5)
        compare_snr(scans_1_5T, scans_3T, axis)
        plt.show()

    # Apply degradation to BraTS scans
    elif action == "batch-convert":
        seq = "t2f"                                 # t1c, t1n, t2f, t2w
        dataset = "BraSyn"                             # BraSyn, GLI

        data_path = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
        paths, validate_paths = get_brats_paths(data_path, seq, dataset)

        real_images = []
        real_kspaces = []
        simulated_kspaces = []
        simulated_images = []

        axis = 0
        for path in tqdm(paths):
            image = read_nifti(path)
            kspace = convert_to_kspace(image)
            simulated_image, simulated_kspace = generate_simulated_image(kspace, axis=axis)

            real_images.append(image)
            real_kspaces.append(kspace)
            simulated_kspaces.append(simulated_kspace)
            simulated_images.append(simulated_image)