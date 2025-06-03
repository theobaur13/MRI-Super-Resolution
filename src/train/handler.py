import os
import torch
from src.train.loop import loop
from torch.utils.data import DataLoader
from src.train.dataset import MRIDataset

def train(args):
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    
    dataset = MRIDataset(data_dir=args.dataset_dir, axis=args.axis)
    print(f"Total slices in dataset: {len(dataset)}")  

    dataloader = DataLoader(dataset, batch_size=26, shuffle=True, pin_memory=True, num_workers=6)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loop(epochs=100, dataloader=dataloader, output_dir=output_dir)