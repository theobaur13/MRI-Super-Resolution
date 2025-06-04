import lmdb
import pickle
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, axis=2, split="train", limit=10000):
        self.lmdb_path = lmdb_path
        self.axis = axis
        self.split = split
        self.limit = limit

        self.pairs = []                 # (HR_path, LR_path, slice_index)

        self.hr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        self._gather_slices()

    def _gather_slices(self):
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                count = 0
                for key, _ in tqdm(cursor):
                    key_str = key.decode("utf-8")
                    if key_str.startswith(f"{self.split}/") and "/HR/" in key_str:
                        lr_key = key_str.replace("/HR/", "/LR/")
                        self.pairs.append((lr_key, key_str))
                        count += 1
                        if self.limit and count >= self.limit:
                            break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_key, hr_key = self.pairs[idx]

        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                lr_slice = pickle.loads(txn.get(lr_key.encode("utf-8")))
                hr_slice = pickle.loads(txn.get(hr_key.encode("utf-8")))

        # Normalize
        lr_slice = self.normalize_slice(lr_slice)
        hr_slice = self.normalize_slice(hr_slice)

        # Convert to image
        lr_img = Image.fromarray((lr_slice * 255).astype(np.uint8))
        hr_img = Image.fromarray((hr_slice * 255).astype(np.uint8))

        # Transform
        lr_tensor = self.lr_transform(lr_img)
        hr_tensor = self.hr_transform(hr_img)

        return lr_tensor, hr_tensor
    
    def normalize_slice(self, slice_2d):
        slice_2d = np.nan_to_num(slice_2d)
        if np.max(slice_2d) == np.min(slice_2d):
            return np.zeros_like(slice_2d)
        norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
        return norm