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
def simluation_pipeline(kspace, axis):
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
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
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
    # > py main.py simulate --path "data/ADNI_NIfTIs/3T/ADNI_002_S_0413_MR_Double_TSE_br_raw_20061115141733_1_S22682_I30117.nii.gz" --axis 0 --slice 24
    elif action == "simulate":
        # Arguments
        axis = args.axis
        slice_idx = args.slice
        arg_path = args.path
        path = os.path.join(base_dir, arg_path)

        path_1_5T, path_3T = get_matching_adni_scan(path)
        
        nifti_1_5T = read_nifti(path_1_5T)
        nifti_3T = read_nifti(path_3T)

        volume_1_5T = jnp.array(nifti_1_5T.get_fdata())
        volume_3T = jnp.array(nifti_3T.get_fdata())

        kspace_1_5T = convert_to_kspace(volume_1_5T)
        kspace_3T = convert_to_kspace(volume_3T)
        simulated_volume, simulated_kspace = simluation_pipeline(kspace_3T, axis=0)
        simulated_nifti = nib.Nifti1Image(simulated_volume, affine=nifti_3T.affine)

        max_value = max(
            robust_max(kspace_1_5T, axis, slice_idx),
            robust_max(kspace_3T, axis, slice_idx),
            robust_max(simulated_kspace, axis, slice_idx)
        ) * slice_idx * 10

        display_comparison([nifti_1_5T, nifti_3T], slice=slice_idx, axis=axis)
        display_comparison([nifti_1_5T, simulated_nifti], slice=slice_idx, axis=axis)
        plt.show()

    # Perform analysis between two types of scans
    # > py main.py analyse-noise --dataset "ADNI" --axis 0 --limit 52
    # > py main.py analyse-noise --dataset "BraTS" --axis 0 --limit 15
    # > py main.py analyse-brightness --dataset "ADNI" --slice 24 --axis 0 --limit 52
    # > py main.py analyse-brightness --dataset "BraTS" --slice 65 --axis 0 --limit 15
    elif action == "analyse-noise" or action == "analyse-brightness":
        # Arguments
        axis = args.axis
        limit = args.limit
        datset = args.dataset.lower()
        shape = (99,117,48)

        if datset == "adni":
            read_fn = lambda path: read_nifti(path).get_fdata()[0:shape[0], 0:shape[1], 0:shape[2]]
            paths_1_5T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "1.5T")))
            paths_3T = sorted(os.listdir(os.path.join(ADNI_nifti_dir, "3T")))
            paths_1_5T = [os.path.join(ADNI_nifti_dir, "1.5T", p) for p in paths_1_5T]
            paths_3T = [os.path.join(ADNI_nifti_dir, "3T", p) for p in paths_3T]

        elif datset == "brats":
            read_fn = lambda path: read_nifti(path).get_fdata()
            train_paths, validate_paths = get_brats_paths(brats_dir, "t2f", "BraSyn")
            paths_1_5T = train_paths + validate_paths  # Placeholder pairing
            paths_3T = train_paths + validate_paths

        # Allocate hypervolumes
        hypervolume_1_5T = np.zeros((limit, *shape))
        hypervolume_3T = np.zeros((limit, *shape))

        # Load volumes
        for i in tqdm(range(limit)):
            hypervolume_1_5T[i] = read_fn(paths_1_5T[i])
            hypervolume_3T[i] = read_fn(paths_3T[i])

        if action == "analyse-noise":
            compare_snr(hypervolume_1_5T, hypervolume_3T, axis)

        elif action == "analyse-brightness":
            slice_idx = args.slice
            generate_brightness_mask(hypervolume_1_5T, hypervolume_3T, slice_idx, axis=axis, sigma=20, lim=0.6)

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
            volume = jnp.array(nifti.get_fdata())
            kspace = convert_to_kspace(volume)
            simulated_volume, _ = simluation_pipeline(kspace, axis=axis)
            simulated_nifti = nib.Nifti1Image(simulated_volume, affine=nifti.affine)

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
        plot_slice(ax, nifti, slice=args.slice, axis=args.axis)
        plt.show()