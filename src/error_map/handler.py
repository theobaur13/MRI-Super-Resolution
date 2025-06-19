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
    slice_index = args.slice
    
    # Load the generator model
    generator = Generator().to("cuda")
    generator.load_state_dict(torch.load(model_path, map_location="cuda"))

    hr_key = f"train/{vol_name}/HR/{slice_index:03d}".encode("utf-8")
    lr_key = f"train/{vol_name}/LR/{slice_index:03d}".encode("utf-8")

    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            hr_slice = pickle.loads(txn.get(hr_key))
            lr_slice = pickle.loads(txn.get(lr_key))

    # Convert LR slice to tensor and run through the generator
    lr_tensor = torch.tensor(lr_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    sr_slice = sr_tensor.squeeze().cpu().numpy()
    hr_slice = hr_slice.squeeze()

    # Calculate the MAE map
    map = (hr_slice - sr_slice) ** 2

    #TODO: Add Sobel edge loss and perceptual loss

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
    plt.imshow(map, cmap='hot', vmin=0, vmax=map.max())
    plt.title('Map')
    plt.colorbar(label='Error')

    plt.tight_layout()
    plt.show()