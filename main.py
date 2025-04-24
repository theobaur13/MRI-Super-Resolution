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
from src.gibbs_removal import *

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    ADNI_dir = os.path.join(data_path, "ADNI")
    ADNI_collapsed_dir = os.path.join(data_path, "ADNI_collapsed")

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["organise-adni", "simulate", "analyse", "batch-convert"], help="Action to perform")
    parser.add_argument("--subject", type=str, help="Subject for analysis (e.g., 'noise', 'brightness')")
    parser.add_argument("--index", type=int, help="Index of ADNI image to simulate")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
    parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    parser.add_argument("--brats_dir", type=str, help="Name of the BraTS directory")
    parser.add_argument("--seq", type=str, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    parser.add_argument("--dataset", type=str, help="Dataset type (e.g., 'BraSyn', 'GLI')")
    parser.add_argument("--output_dir", type=str, help="Name of the output directory")
    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    # > py main.py organise-adni
    if action == "organise-adni":
        collapse_adni(ADNI_dir, ADNI_collapsed_dir)

    # Apply degradation to ADNI scans
    # > py main.py simulate --index 0 --axis 0 --slice 24
    elif action == "simulate":
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
        df = adni_dataframe(ADNI_collapsed_dir)
        subject = args.subject
        axis = args.axis
        limit = args.limit

        # Read ADNI scans
        hypervolume_1_5T = np.zeros((limit, 48, 256, 256))
        hypervolume_3T = np.zeros((limit, 48, 256, 256))
        for i in tqdm(range(limit)):
            paths_1_5T, paths_3T = get_adni_pair(df, i)
            image_1_5T = read_dicom(paths_1_5T)
            image_1_5T = image_1_5T[0:48, 0:256, 0:256]

            image_3T = read_dicom(paths_3T, flip=True)
            image_3T = image_3T[0:48, 0:256, 0:256]

            hypervolume_1_5T[i] = image_1_5T
            hypervolume_3T[i] = image_3T

        # > py main.py analyse --subject "noise" --axis 0 --limit 5
        if subject == "noise":
            compare_snr(hypervolume_1_5T, hypervolume_3T, axis)

        # > py main.py analyse --subject "brightness" --slice 24 --axis 0 --limit 5
        elif subject == "brightness":
            slice_idx = args.slice
            generate_brightness_mask(hypervolume_1_5T, hypervolume_3T, slice_idx, axis=axis, sigma=5)

        # > py main.py analyse --subject "gibbs" --index 0 --slice 24 --axis 0 --limit 5
        elif subject == "gibbs":
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
    # > py main.py batch-convert --brats_dir "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000" --seq "t2f" --dataset "BraSyn" --output_dir "BraTS_output"
    elif action == "batch-convert":
        seq = args.seq                  # t1c, t1n, t2f, t2w
        dataset = args.dataset          # BraSyn, GLI
        brats_dir = args.brats_dir      # data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000
        output_dir = args.output_dir    # BraTS_output
        
        brats_path = os.path.join(data_path, brats_dir)
        output_path = os.path.join(data_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        paths, validate_paths = get_brats_paths(brats_path, seq, dataset)

        axis = 0
        for path in tqdm(paths):
            image = read_nifti(path)
            kspace = convert_to_kspace(image)
            simulated_image, simulated_kspace = generate_simulated_image(kspace, axis=axis)

            write_nifti(simulated_image, os.path.join(output_path, os.path.basename(path)))