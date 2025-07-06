import os
import torch
import lmdb
import sys
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train.models.ESRGAN import Generator

# For a specific loss type generate a map of L1 loss showing the difference between the pre-trained model and the model after training for 4 epochs.
def main():
    # Load the pre-trained and trained models
    loss_type = "fourier"                   # Change this to "pixel", "perceptual", "edge", or "style" as needed
    epoch = 4
    trained_model_path = f"E:/ESRGAN_RRDB3_{loss_type}_50-100_20k/generator_epoch_{epoch}.pth"
    pretrained_model_path = f"E:/ESRGAN_RRDB3_{loss_type}_50-100_20k/pretrain_epoch_4.pth"

    pretrained_generator = Generator(rrdb_count=3).to("cuda")
    trained_generator = Generator(rrdb_count=3).to("cuda")

    pretrained_generator.load_state_dict(torch.load(pretrained_model_path))
    trained_generator.load_state_dict(torch.load(trained_model_path))

    # Load a sample input image (e.g., from the training dataset)
    lmdb_path = os.getenv("LMDB_PATH")
    slice_path = "validate/BraTS-GLI-00213-000-t2f/HR/065"

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        slice_data = txn.get(slice_path.encode())
    
    hr_slice = pickle.loads(gzip.decompress(slice_data))
    hr_slice = (hr_slice - np.min(hr_slice)) / (np.max(hr_slice) - np.min(hr_slice))

    hr_image = torch.tensor(hr_slice).unsqueeze(0).unsqueeze(0).float().to("cuda")

    # Generate super-resolved images using both models
    with torch.no_grad():
        sr_pretrained = pretrained_generator(hr_image)
        sr_trained = trained_generator(hr_image)

    # Calculate the L1 loss between the super-resolved images
    l1_loss = torch.abs(sr_pretrained - sr_trained).squeeze().cpu().numpy()

    # Plot the L1 loss map
    plt.figure(figsize=(10, 10))
    plt.imshow(l1_loss, cmap='hot', vmin=0, vmax=0.5)
    plt.colorbar(label='L1 Loss')
    plt.title(f'L1 Loss Map ({loss_type} loss)')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()