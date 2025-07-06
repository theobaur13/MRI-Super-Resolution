import os
import torch
from src.train.loop import GAN_loop, CNN_loop

from torch.utils.data import DataLoader
from src.train.dataset import LMDBDataset

def train(args):
    print(f"Training on device: {torch.cuda.get_device_name(0)}")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constants
    BATCH_SIZE = 4
    EPOCHS = 20
    PRETRAIN_EPOCHS = 4
    NUM_WORKERS = 4
    USEFUL_RANGE = (50, 100)
    LIMIT = 20000
    RRDB_COUNT = 3

    train_data = LMDBDataset(lmdb_path=args.lmdb_path,split="train", limit=LIMIT, useful_range=USEFUL_RANGE)
    val_data = LMDBDataset(lmdb_path=args.lmdb_path, split="validate", limit=LIMIT * 0.33, useful_range=USEFUL_RANGE)
    print(f"Total slices in training dataset: {len(train_data)}")
    print(f"Total slices in validation dataset: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    GAN_loop(train_loader, val_loader, epochs=EPOCHS, pretrain_epochs=PRETRAIN_EPOCHS, rrdb_count=RRDB_COUNT, output_dir=output_dir, resume=args.resume)
    CNN_loop(train_loader, val_loader, epochs=EPOCHS, output_dir=output_dir, resume=args.resume)