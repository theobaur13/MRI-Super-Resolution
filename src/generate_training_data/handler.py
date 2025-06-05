import os
import lmdb
import pickle
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from src.simulation.pipeline import simulate_batch
from src.utils.conversions import jax_to_numpy
from src.utils.readwrite import read_nifti
from src.utils.paths import get_brats_paths

def generate_training_data(args):
    # Arguments
    brats_dir = args.brats_dir                              # Path to BraTS directory
    output_dir = args.output_dir                            # Output directory for converted data
    limit = args.limit
    axis = args.axis
    seq = args.seq
    batch_size = 4
    useful_range = (20, 140)

    map_size = int(50 * 1024 * 1024 * 1024)  # 50 GB
    env = lmdb.open(output_dir, map_size=map_size)

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

        # Read images
        for j in range(i, min(i + batch_size, train_limit)):
            nifti = read_nifti(train_paths[j])
            train_images.append(nifti.get_fdata())
        
        # Stack images into for parallel processing
        batch_np = np.stack(train_images).astype(np.float32)
        batch_jax = jnp.array(batch_np)

        # Simulate batches of images
        batch_results, _ = simulate_batch(batch_jax, axis)

        # Save simulated images
        for j in range(batch_results["final"].shape[0]):
            vol_id = os.path.basename(train_paths[i + j]).replace(".nii.gz", "")
            hr_vol = batch_np[j]
            lr_vol = jax_to_numpy(batch_results["final"][j])

            with env.begin(write=True) as txn:
                for s in range(useful_range[0], useful_range[1]):
                    hr_slice = np.take(hr_vol, s, axis=axis).astype(np.float32)
                    lr_slice = np.take(lr_vol, s, axis=axis).astype(np.float32)

                    hr_key = f"train/{vol_id}/HR/{s:03d}".encode()
                    lr_key = f"train/{vol_id}/LR/{s:03d}".encode()

                    txn.put(hr_key, pickle.dumps(hr_slice))
                    txn.put(lr_key, pickle.dumps(lr_slice))

    print(f"Simulating {validate_limit} validation scans")
    for i in tqdm(range(0, validate_limit, batch_size)):
        validate_images = []

        # Read images
        for j in range(i, min(i + batch_size, validate_limit)):
            nifti = read_nifti(validate_paths[j])
            validate_images.append(nifti.get_fdata())
        
        # Stack images into for parallel processing
        batch_np = np.stack(validate_images).astype(np.float32)
        batch_jax = jnp.array(batch_np)

        # Simulate batches of images
        batch_results, _ = simulate_batch(batch_jax, axis)

        # Save simulated images
        for j in range(batch_results["final"].shape[0]):
            vol_id = os.path.basename(validate_paths[i + j]).replace(".nii.gz", "")
            hr_vol = batch_np[j]
            lr_vol = jax_to_numpy(batch_results["final"][j])

            with env.begin(write=True) as txn:
                for s in range(useful_range[0], useful_range[1]):
                    hr_slice = np.take(hr_vol, s, axis=axis).astype(np.float32)
                    lr_slice = np.take(lr_vol, s, axis=axis).astype(np.float32)

                    hr_key = f"validate/{vol_id}/HR/{s:03d}".encode()
                    lr_key = f"validate/{vol_id}/LR/{s:03d}".encode()

                    txn.put(hr_key, pickle.dumps(hr_slice))
                    txn.put(lr_key, pickle.dumps(lr_slice))
    
    env.sync()
    env.close()
    print(f"LMDB dataset written to {output_dir}")