import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.train.models.ESRGAN import Generator
from src.train.loss import CompositeLoss
from src.train.dataset import LMDBDataset
from torch.utils.data import DataLoader

# Log raw loss magnitudes during initial forward pass to calibrate loss weights
def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    LMDB_PATH = os.getenv("LMDB_PATH")

    generator = Generator().to("cuda")
    loss_fn = CompositeLoss(weights={"edge": 0.7, "pixel": 0.3, "perceptual": 1.0, "fourier": 1.0, "style": 0.0}).to("cuda")

    train_data = LMDBDataset(lmdb_path=LMDB_PATH, split="train", limit=1000, useful_range=(30, 150))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    loss_log = {
        "pixel": [],
        "perceptual": [],
        "edge": [],
        "fourier": [],
        "style": []
    }

    for lr, hr in tqdm(train_loader):
        lr, hr = lr.to("cuda"), hr.to("cuda")

        with torch.no_grad():
            sr = generator(lr)

        # Calculate losses
        _, losses = loss_fn(sr, hr, logging=True)

        loss_log["pixel"].append(losses["pixel"].item())
        loss_log["perceptual"].append(losses["perceptual"].item())
        loss_log["edge"].append(losses["edge"].item())
        loss_log["fourier"].append(losses["fourier"].item())
        # loss_log["style"].append(losses["style"].item())

    # Plotting the loss magnitudes
    plt.figure(figsize=(10, 6))
    for key, values in loss_log.items():
        plt.plot(values, label=key)

    plt.xlabel("Sample Index")
    plt.ylabel("Loss Magnitude")
    plt.title("Raw Loss Magnitudes per Sample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()