import os
import subprocess
import shutil
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import dicom2nifti
import nibabel as nib
from src.pipeline import simluation_pipeline, simulate_batch
from src.conversions import convert_to_kspace, jax_to_numpy
from src.readwrite import read_nifti, write_nifti
from src.display import display_img, display_3d
from src.analysis import compare_snr, generate_brightness_map, generate_snr_map
from src.slicing import slice_nifti
from src.utils import get_adni_paths, get_matching_adni_scan, get_brats_paths

def convert_adni(args):
    ADNI_dir = args.ADNI_dir
    output_dir = args.ADNI_nifti_dir

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

def simulate(args):
    # Arguments
    axis = args.axis
    slice_idx = args.slice
    path = args.path
    compare = args.compare

    if compare:
        # Get the matching 1.5T and 3T scans
        path_1_5T, path_3T = get_matching_adni_scan(path)
        nifti_1_5T = read_nifti(path_1_5T)
    else:
        path_3T = path
    
    nifti_3T = read_nifti(path_3T)

    simulated_nifti = simluation_pipeline(nifti_3T, axis, path, visualize=True, slice=slice_idx)

    if compare:
        # Display the target vs simulated image
        display_img([nifti_1_5T, simulated_nifti], slice=slice_idx, axis=axis, titles=["Original 1.5T Image", "Simulated 1.5T Image"])
    else:
        # Display the simulated image
        display_img([nifti_3T, simulated_nifti], slice=slice_idx, axis=axis, titles=["Original 3T Image", "Simulated 1.5T Image"])    
    plt.show()

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

def generate_training_data(args):
    # Arguments
    brats_dir = args.brats_dir                              # Path to BraTS directory
    output_dir = args.output_dir                            # Output directory for converted data
    limit = args.limit
    axis = args.axis
    batch_size = 5

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validate"), exist_ok=True)

    train_paths, validate_paths = get_brats_paths(brats_dir)

    if not limit:
        train_limit = len(train_paths)
        validate_limit = len(validate_paths)
    else:
        train_limit = min(limit, len(train_paths))
        # Keep the same ratio of train to validate as in the original dataset
        validate_limit = int(train_limit * (len(validate_paths) / len(train_paths)))

    print(f"Simulating {train_limit} training scans and {validate_limit} validation scans...")
    
    for i in tqdm(range(0, train_limit, batch_size)):
        train_images = []
        train_affines = []

        for j in range(i, min(i + batch_size, train_limit)):
            nifti = read_nifti(train_paths[j])
            train_images.append(nifti.get_fdata())
            train_affines.append(nifti.affine)
        
        print("Stacking images into numpy arrays...")
        batch_images_np = np.stack(train_images).astype(np.float32)
        batch_affines = np.array(train_affines)
        batch_images_jax = jnp.array(batch_images_np)

        print("Simulating batches of images...")
        batch_results, _ = simulate_batch(batch_images_jax, axis)

        print("Saving simulated images...")
        for j in range(batch_results["final"].shape[0]):
            final_image = batch_results["final"][j]
            affine = batch_affines[j]
            out_path = os.path.join(output_dir, "train", os.path.basename(train_paths[i + j]))
            write_nifti(nib.Nifti1Image(jax_to_numpy(final_image), affine), out_path)

def view(args):
    nifti = read_nifti(args.path, normalise=False)
    display_img([nifti], slice=args.slice, axis=args.axis)
    plt.show()

def segment(args):
    paths, _ = get_brats_paths(args.dataset_dir)

    flywheel_dir = os.path.join(os.getcwd(), "flywheel", "v0")
    flywheel_input_dir = os.path.join(flywheel_dir, "input", "nifti")
    flywheel_output_dir = os.path.join(flywheel_dir, "output")
    config_path = os.path.join(flywheel_dir, "config.json")

    for i in tqdm(range(args.limit)):
        # Skip condition
        scan_path = paths[i]
        scan_dir = os.path.dirname(scan_path)
        scan_name = os.path.basename(scan_path).split(".")[0]
        target = scan_name + "_fast_mixeltype.nii.gz"
        seq = scan_name.split("-")[-1]
        if seq in ["t1c", "t1n"]:
            n = "1"
        elif seq in ["t2f", "t2w"]:
            n = "2"
            
        print(f"Segmenting {scan_path} with sequence {seq}...")
        if os.path.exists(os.path.join(scan_dir, target)):
            print(f"Skipping {scan_path} as it has already been segmented.")
            continue

        shutil.copy(scan_path, flywheel_input_dir)

        subprocess.run([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{config_path}:/flywheel/v0/config.json",
            "-v", f"{flywheel_input_dir}:/flywheel/v0/input/nifti",
            "-v", f"{flywheel_output_dir}:/flywheel/v0/output",
            "scitran/fsl-fast",
            "-t", n
        ])

        for file in os.listdir(flywheel_output_dir):
            if file.endswith(".nii.gz"):
                shutil.copy(os.path.join(flywheel_output_dir, file), os.path.join(scan_dir, file))

        # Clear input and output directories
        os.remove(os.path.join(flywheel_input_dir, os.path.basename(scan_path)))
        for file in os.listdir(flywheel_output_dir):
            if file != '.gitkeep':
                os.remove(os.path.join(flywheel_output_dir, file))

def train(args):
    dataset_dir = args.dataset_dir
    
    