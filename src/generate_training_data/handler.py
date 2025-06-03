import os
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import nibabel as nib
import shutil
from src.simulation.pipeline import simulate_batch
from src.utils.conversions import jax_to_numpy
from src.utils.readwrite import read_nifti, write_nifti
from src.utils.paths import get_brats_paths

def generate_training_data(args):
    # Arguments
    brats_dir = args.brats_dir                              # Path to BraTS directory
    output_dir = args.output_dir                            # Output directory for converted data
    limit = args.limit
    axis = args.axis
    seq = args.seq
    batch_size = 5

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validate"), exist_ok=True)

    train_paths, validate_paths = get_brats_paths(brats_dir, seq)

    if not limit:
        train_limit = len(train_paths)
        validate_limit = len(validate_paths)
    else:
        train_limit = min(limit, len(train_paths))
        # Keep the same ratio of train to validate as in the original dataset
        validate_limit = int(train_limit * (len(validate_paths) / len(train_paths)))

    print(f"Simulating {train_limit} training scans")
    for i in tqdm(range(0, train_limit, batch_size)):
        train_images = []
        train_affines = []

        # Read images
        for j in range(i, min(i + batch_size, train_limit)):
            nifti = read_nifti(train_paths[j])
            train_images.append(nifti.get_fdata())
            train_affines.append(nifti.affine)
        
        # Stack images into for parallel processing
        batch_images_np = np.stack(train_images).astype(np.float32)
        batch_images_jax = jnp.array(batch_images_np)

        # Simulate batches of images
        batch_results, _ = simulate_batch(batch_images_jax, axis)

        # Save simulated images
        for j in range(batch_results["final"].shape[0]):
            final_image = batch_results["final"][j]

            # Save simulated image to output directory as _LR.nii.gz
            simulated_filename = os.path.basename(train_paths[i + j]).replace(".nii.gz", f"_LR.npy")
            destination = os.path.join(output_dir, "train", simulated_filename)
            print(f"Saving simulated image to {destination}")
            np.save(destination, jax_to_numpy(final_image))

            # Copy original image to output directory as _HR.nii.gz with shutil
            original_filename = os.path.basename(train_paths[i + j]).replace(".nii.gz", f"_HR.npy")
            destination = os.path.join(output_dir, "train", original_filename)
            np.save(destination, train_images[j])

    print(f"Simulating {validate_limit} validation scans")
    for i in tqdm(range(0, validate_limit, batch_size)):
        validate_images = []
        validate_affines = []

        # Read images
        for j in range(i, min(i + batch_size, validate_limit)):
            nifti = read_nifti(validate_paths[j])
            validate_images.append(nifti.get_fdata())
            validate_affines.append(nifti.affine)
        
        # Stack images into for parallel processing
        batch_images_np = np.stack(validate_images).astype(np.float32)
        batch_images_jax = jnp.array(batch_images_np)

        # Simulate batches of images
        batch_results, _ = simulate_batch(batch_images_jax, axis)

        # Save simulated images
        for j in range(batch_results["final"].shape[0]):
            final_image = batch_results["final"][j]

            # Save simulated image to output directory as _LR.nii.gz
            simulated_filename = os.path.basename(train_paths[i + j]).replace(".nii.gz", f"_LR.npy")
            destination = os.path.join(output_dir, "validate", simulated_filename)
            np.save(destination, jax_to_numpy(final_image))

            # Copy original image to output directory as _HR.nii.gz with shutil
            original_filename = os.path.basename(train_paths[i + j]).replace(".nii.gz", f"_HR.npy")
            destination = os.path.join(output_dir, "validate", original_filename)
            np.save(destination, validate_images[j])