import torch
import lmdb
import pickle
from src.train.models.ESRGAN import Generator
import matplotlib.pyplot as plt

def error_map(args):
    # Handle command line arguments
    model_path = args.model_path
    lmdb_path = args.lmdb_path
    vol_name = args.vol_name
    axis = args.axis
    slice_index = args.slice
    
    # Load the generator model
    generator = Generator().to("cuda")
    generator.load_state_dict(torch.load(model_path, map_location="cuda"))

    # Retrieve the HR and LR slices from the LMDB database
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

    # Convert LR slice to tensor and run through the generator
    lr_tensor = torch.tensor(lr_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    sr_slice = sr_tensor.squeeze().cpu().numpy()
    hr_slice = hr_slice.squeeze()

    # Calculate the MSE map
    mse_map = (hr_slice - sr_slice) ** 2

    # Plotting the MSE heatmap using mse_map
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(lr_slice.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('Low Resolution Slice')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sr_slice, cmap='gray', vmin=0, vmax=1)
    plt.title('Super Resolution Slice')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hr_slice, cmap='gray', vmin=0, vmax=1)
    plt.title('High Resolution Slice')
    plt.axis('off')

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(mse_map, cmap='hot', vmin=0, vmax=mse_map.max())
    plt.title('MSE Map')
    plt.colorbar(label='Mean Squared Error')

    plt.tight_layout()
    plt.show()        