import torch
from src.train.loop import loop
from torch.utils.data import DataLoader
from src.train.dataset import MRIDataset

def train(args):
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    dataset = MRIDataset(data_dir=args.dataset_dir, axis=args.axis)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    loop(epochs=100, dataloader=dataloader)