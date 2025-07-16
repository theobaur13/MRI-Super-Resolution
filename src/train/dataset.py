import lmdb
import pickle
import gzip
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split, limit, useful_range):
        self.lmdb_path = lmdb_path
        self.split = split
        self.limit = limit
        self.useful_range = useful_range

        self.do_augment = split == "train"
        self.pairs = []                 # (HR_path, LR_path, slice_index)

        self.hr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self._gather_slices()

    def _gather_slices(self):
        prefix = f"{self.split}/".encode("utf-8")
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()

                if not cursor.set_range(prefix):
                    return
                
                count = 0
                for key, _ in tqdm(cursor):
                    key_str = key.decode("utf-8")
                    if key_str.startswith(f"{self.split}/") and "/HR/" in key_str:
                        # Get the slice index
                        try:
                            slice_idx = int(key_str.rsplit("/", 1)[-1])
                        except ValueError:
                            continue

                        # Filter by useful_range
                        if not (self.useful_range[0] <= slice_idx < self.useful_range[1]):
                            continue
                        
                        # Check if HR slice contains only zeroes
                        hr_bytes = txn.get(key)
                        if hr_bytes is None:
                            continue

                        hr_slice = pickle.loads(gzip.decompress(hr_bytes))
                        if np.max(hr_slice) == np.min(hr_slice) or np.std(hr_slice) < 1e-5:
                            continue

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
                lr_slice = pickle.loads(gzip.decompress(txn.get(lr_key.encode("utf-8"))))
                hr_slice = pickle.loads(gzip.decompress(txn.get(hr_key.encode("utf-8"))))

        # Normalize
        lr_slice = self.normalize_slice(lr_slice)
        hr_slice = self.normalize_slice(hr_slice)

        # Convert to image
        lr_img = Image.fromarray((lr_slice * 255).astype(np.uint8))
        hr_img = Image.fromarray((hr_slice * 255).astype(np.uint8))

        # Transform
        lr_tensor = self.lr_transform(lr_img)
        hr_tensor = self.hr_transform(hr_img)

        # Close the images to free memory
        lr_img.close()
        hr_img.close()

        # Augmentation
        if self.do_augment:
            lr_tensor, hr_tensor = self.augment_pair(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor
    
    def normalize_slice(self, slice_2d):
        slice_2d = np.nan_to_num(slice_2d)
        if np.max(slice_2d) == np.min(slice_2d):
            return np.zeros_like(slice_2d)
        norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
        return norm
    
    def augment_pair(self, lr_img, hr_img):
        # Random horizontal flip
        if random.random() > 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)

        # Random vertical flip
        if random.random() > 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)

        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)

        return lr_img, hr_img