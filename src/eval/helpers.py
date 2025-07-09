from tqdm import tqdm

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