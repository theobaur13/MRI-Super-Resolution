import os
import lmdb
import pickle
import gzip
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from src.simulation.pipeline import simulate_batch
from src.utils.conversions import jax_to_numpy
from src.utils.readwrite import read_nifti
from src.utils.paths import get_brats_paths

def generate_training_data(args):
    brats_dir = args.brats_dir
    output_dir = args.output_dir
    limit = args.limit
    axis = args.axis
    seq = args.seq
    normalise = args.normalise
    batch_size = 8
    useful_range = (1, 150)
    map_size = int(60 * 1024 * 1024 * 1024)
    # map_size = int(20 * 1024 * 1024 * 1024)  # 20 GB

    env = lmdb.open(output_dir, map_size=map_size)
    train_paths, validate_paths, test_paths = get_brats_paths(brats_dir, seq, dataset="BraSyn")

    def process_split(split_name, paths, limit):
        print(f"Simulating {limit} {split_name} scans")
        for i in tqdm(range(0, limit, batch_size)):
            images, ids = [], []
            for j in range(i, min(i + batch_size, limit)):
                path = paths[j]
                vol_id = os.path.basename(path).replace(".nii.gz", "")
                if volume_already_processed(env, vol_id, split=split_name):
                    continue
                images.append(read_nifti(path, normalise).get_fdata())
                ids.append(vol_id)

            if not images:
                continue

            batch_np = np.stack(images).astype(np.float32)
            batch_jax = jnp.array(batch_np)
            batch_results, _ = simulate_batch(batch_jax, axis)

            for idx, vol_id in tqdm(enumerate(ids), total=len(ids), leave=False):
                hr_vol = batch_np[idx]
                lr_vol = jax_to_numpy(batch_results["final"][idx])

                with env.begin(write=True) as txn:
                    for s in tqdm(range(*useful_range), leave=False):
                        hr_slice = np.take(hr_vol, s, axis=axis).astype(np.float32)
                        lr_slice = np.take(lr_vol, s, axis=axis).astype(np.float32)

                        hr_key = f"{split_name}/{vol_id}/HR/{s:03d}".encode()
                        lr_key = f"{split_name}/{vol_id}/LR/{s:03d}".encode()
                        txn.put(hr_key, gzip.compress(pickle.dumps(hr_slice)))
                        txn.put(lr_key, gzip.compress(pickle.dumps(lr_slice)))

    # Set train/validate limits
    if not limit:
        train_limit = len(train_paths)
        validate_limit = len(validate_paths)
        test_limit = len(test_paths)
    else:
        train_limit = min(limit, len(train_paths))
        ratio = len(validate_paths) / len(train_paths)
        validate_limit = int(train_limit * ratio)

    process_split("train", train_paths, train_limit)
    process_split("validate", test_paths, test_limit)
    process_split("test", validate_paths, validate_limit)

    env.sync()
    env.close()
    print(f"LMDB dataset written to {output_dir}")

def volume_already_processed(env, vol_id, split="train", slice_check=20):
    key = f"{split}/{vol_id}/HR/{slice_check:03d}".encode()
    with env.begin(write=False) as txn:
        return txn.get(key) is not None