import os
import sys
from src.utils import *
from src.simulation import *
from src.data_routing import *

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    # Command line arguments for dataset type
    if len(sys.argv) < 2:
        print("Usage: python main.py [action]")
        print("action: 'organise-adni' or 'simulate' or 'batch-convert'")
        sys.exit(1)
    action = sys.argv[1].lower()

    if action == "organise-adni":
        ADNI_dir = os.path.join(data_path, "ADNI")
        new_dir = os.path.join(data_path, "ADNI_collapsed")
        collapse_adni(ADNI_dir, new_dir)

    elif action == "simulate":
        if len(sys.argv) < 3:
            print("Usage: python main.py simulate [index]")
            sys.exit(1)

        index = sys.argv[1].lower()
        dir = os.path.join(data_path, "ADNI_collapsed")
        df = adni_dataframe(dir)
        paths_1_5T, paths_3T = get_adni_pair(df, index)
        
        image_T1_5 = read_dicom(paths_1_5T)

        image_T3 = read_dicom(paths_3T)
        image_T3 = image_T3[0:48, 0:256, 0:256]

        T1_5_kspace = convert_to_kspace(image_T1_5)
        T3_kspace = convert_to_kspace(image_T3)
        simulated_image, simulated_kspace = generate_simulated_image(T3_kspace, axis=0)

        slice_idx = 24
        display_comparison(image_T1_5, image_T3, slice=slice_idx, axis=0, kspace=False)
        display_comparison(image_T1_5, simulated_image, slice=slice_idx, axis=0, kspace=False)
        plot_3d_kspace([T1_5_kspace, T3_kspace, simulated_kspace], slice_idx, axis=0, cmap="viridis")
        plt.show()

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