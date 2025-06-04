import os
import torch
from src.train.loop import loop
from torch.utils.data import DataLoader
from src.train.dataset import LMDBDataset

def train(args):
    print(f"Training on device: {torch.cuda.get_device_name(0)}")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constants
    BATCH_SIZE = 20         # NOTE: Anything over 20 dips into normal RAM over VRAM
    EPOCHS = 15
    NUM_WORKERS = 4
    LIMIT = 20000           # NOTE: Set to None for no limit

    dataset = LMDBDataset(lmdb_path=args.lmdb_path, axis=args.axis, limit=LIMIT)
    print(f"Total slices in dataset: {len(dataset)}")  

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

    loop(epochs=EPOCHS, dataloader=dataloader, output_dir=output_dir)