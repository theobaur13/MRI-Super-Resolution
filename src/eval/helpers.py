import lmdb
import torch
from tqdm import tqdm
from src.utils.inference import load_model, run_model_on_slice

def group_slices(slices):
    grouped = {}
    for slice_key in slices:
        parts = slice_key.split('/')
        vol_id = parts[1]  # Extract volume ID
        if vol_id not in grouped:
            grouped[vol_id] = []
        grouped[vol_id].append(slice_key)

    # Order the slices by their index
    for vol_id, slice_keys in grouped.items():
        slice_keys.sort(key=lambda x: int(x.split("/")[-1]))
    return grouped

def get_LMDB_validate_paths(env):
    validate_prefix = b"validate/"
    print("Retrieving validation LR slice paths...")
    with env.begin() as txn:
        cursor = txn.cursor()
        validation_paths = []
        if cursor.set_range(validate_prefix):
            for key, _ in tqdm(cursor):
                if key.startswith(validate_prefix):
                    validation_paths.append(key.decode("utf-8"))
    return validation_paths

def get_grouped_validation_slices(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    validation_paths = get_LMDB_validate_paths(env)
    lr_paths = [p for p in validation_paths if "LR" in p]
    grouped_lr_paths = group_slices(lr_paths)
    return grouped_lr_paths

def generate_SR_HR(model_path, lmdb_path):
    grouped_lr_paths = get_grouped_validation_slices(lmdb_path)
    print(f"Found {len(grouped_lr_paths)} validation volumes.")

    # Load the model
    model = load_model(model_path)
    model.eval()

    # Process each volume's LR slices
    with torch.no_grad():
        for volume, slice_keys in grouped_lr_paths.items():
            for lr_slice_key in slice_keys:
                slice_index = int(lr_slice_key.split("/")[-1])
                sr_slice, hr_slice, _ = run_model_on_slice(
                    model=model,
                    lmdb_path=lmdb_path,
                    vol_name=volume,
                    set_type="validate",
                    slice_index=slice_index,
                )

                # Yield as PyTorch tensors
                yield (
                    torch.tensor(sr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    torch.tensor(hr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )