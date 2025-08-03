import os
import lmdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.inference import load_model, run_model_on_slice
from src.utils.plot import plot_lr_sr_hr

def export_predictions(args):
    output_dir = args.output_dir
    lmdb_path = args.lmdb_path
    model_path = args.model_path
    rrdb_count = args.rrdb_count
    set_type = "validate"

    os.makedirs(output_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        prefix = f"{set_type}/".encode("utf-8")

        # Efficiently collect only HR slice keys for "validate" set
        validate_hr_keys = []
        if cursor.set_range(prefix):
            for key, _ in tqdm(cursor):
                if not key.startswith(prefix):
                    break
                if b"/HR/" in key:
                    validate_hr_keys.append(key.decode("utf-8"))

        # Extract unique volume names
        vol_names = sorted({key.split("/")[1] for key in validate_hr_keys})
        if not vol_names:
            print(f"No {set_type} volumes found in LMDB.")
            return

        # Use first volume to determine slice indices
        first_vol = vol_names[0]
        slice_indices = sorted({
            int(key.split("/")[-1])
            for key in validate_hr_keys
            if f"/{first_vol}/" in key
        })

        print(f"Found {len(vol_names)} volumes and {len(slice_indices)} slices per volume.")

    # Load the model
    model = load_model(model_path, rrdb_count=rrdb_count)

    for vol_name in tqdm(vol_names, desc="Volumes"):
        # Temporarily limit to a specific volume for testing
        if vol_name == "BraTS-GLI-00082-000-t2f":
            break

        vol_output_dir = os.path.join(output_dir, vol_name)
        os.makedirs(vol_output_dir, exist_ok=True)

        for slice_index in tqdm(slice_indices, desc=f"Slices ({vol_name})", leave=False):
            save_path = os.path.join(vol_output_dir, f"{slice_index:03d}.png")
            if os.path.exists(save_path):
                continue  

            try:
                sr_slice, hr_slice, lr_slice = run_model_on_slice(
                    model=model,
                    lmdb_path=lmdb_path,
                    vol_name=vol_name,
                    set_type=set_type,
                    slice_index=slice_index
                )

                fig = plot_lr_sr_hr(lr_slice, sr_slice, hr_slice)
                save_path = os.path.join(vol_output_dir, f"{slice_index:03d}.png")
                fig.savefig(save_path)
                plt.close(fig)

            except Exception as e:
                print(f"[ERROR] Volume: {vol_name}, Slice: {slice_index:03d} -> {e}")