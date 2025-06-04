import torch
import lmdb
import pickle
from src.train.models.ESRGAN import Generator
import matplotlib.pyplot as plt

def run_model(args):
    model_path = args.model_path
    lmdb_path = args.lmdb_path
    vol_name = args.vol_name
    axis = args.axis
    slice_index = args.slice

    generator = Generator().to("cuda")
    generator.load_state_dict(torch.load(model_path, map_location="cuda"))

    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("utf-8")
                if key_str.startswith(f"train/{vol_name}/HR/") and f"/{slice_index:03d}" in key_str:
                    hr_slice = pickle.loads(value)
                if key_str.startswith(f"train/{vol_name}/LR/") and f"/{slice_index:03d}" in key_str:
                    lr_slice = pickle.loads(value)
                    break

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