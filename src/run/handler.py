import torch
import lmdb
import pickle
import gzip
from src.train.models.ESRGAN import Generator
import matplotlib.pyplot as plt

def run_model(args):
    model_path = args.model_path
    lmdb_path = args.lmdb_path
    vol_name = args.vol_name
    set_type = args.set
    slice_index = args.slice
    rrdb_count = args.rrdb_count

    generator = Generator(rrdb_count=rrdb_count).to("cuda")
    print(f"Loading model from {model_path}...")
    generator.load_state_dict(torch.load(model_path, map_location="cuda"))

    hr_key = f"{set_type}/{vol_name}/HR/{slice_index:03d}".encode("utf-8")
    lr_key = f"{set_type}/{vol_name}/LR/{slice_index:03d}".encode("utf-8")

    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            hr_slice = pickle.loads(gzip.decompress(txn.get(hr_key)))
            lr_slice = pickle.loads(gzip.decompress(txn.get(lr_key)))

    lr_tensor = torch.tensor(lr_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    sr_slice = sr_tensor.squeeze().cpu().numpy()
    hr_slice = hr_slice.squeeze()

    # Plotting the slices
    vmin = min(lr_slice.min(), sr_slice.min(), hr_slice.min())
    vmax = max(lr_slice.max(), sr_slice.max(), hr_slice.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_slice.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Low Resolution Slice')
    axes[0].axis('off')

    axes[1].imshow(sr_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Super Resolution Slice')
    axes[1].axis('off')

    axes[2].imshow(hr_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('High Resolution Slice')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()