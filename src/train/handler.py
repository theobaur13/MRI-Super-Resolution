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

    train_data = LMDBDataset(lmdb_path=args.lmdb_path, axis=args.axis,split="train", limit=LIMIT)
    val_data = LMDBDataset(lmdb_path=args.lmdb_path, axis=args.axis, split="validate", limit=LIMIT * 0.25)
    print(f"Total slices in training dataset: {len(train_data)}")
    print(f"Total slices in validation dataset: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    loop(train_loader, val_loader, epochs=EPOCHS, output_dir=output_dir, resume=args.resume)