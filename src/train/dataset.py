import os
import nibabel as nib
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, data_dir, axis=2):
        self.data_dir = data_dir
        self.axis = axis
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
        files = sorted(os.listdir(self.data_dir))

        hr_files = [file for file in files if "_HR" in file]
        for hr_file in hr_files:
            base_name = hr_file.replace("_HR.nii.gz", "")
            lr_file = f"{base_name}_LR.nii.gz"
            hr_path = os.path.join(self.data_dir, hr_file)
            lr_path = os.path.join(self.data_dir, lr_file)

            if os.path.exists(lr_path):
                hr_volume = nib.load(hr_path).get_fdata()
                lr_vol = nib.load(lr_path).get_fdata()

                assert hr_volume.shape == lr_vol.shape, f"Shape mismatch: {hr_volume.shape} vs {lr_vol.shape}"
                
                for slice_index in range(hr_volume.shape[self.axis]):
                    self.pairs.append((hr_path, lr_path, slice_index))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path, lr_path, slice_idx = self.pairs[idx]

        hr_volume = nib.load(hr_path).get_fdata()
        lr_volume = nib.load(lr_path).get_fdata()

        hr_slice = hr_volume.take(indices=slice_idx, axis=self.axis)
        lr_slice = lr_volume.take(indices=slice_idx, axis=self.axis)

        hr_slice = self.normalize_slice(hr_slice)
        lr_slice = self.normalize_slice(lr_slice)

        hr_img = Image.fromarray((hr_slice * 255).astype(np.uint8))
        lr_img = Image.fromarray((lr_slice * 255).astype(np.uint8))

        hr_tensor = self.hr_transform(hr_img)  # [1, 256, 256]
        lr_tensor = self.lr_transform(lr_img)  # [1, 64, 64]

        return lr_tensor, hr_tensor
    
    def normalize_slice(self, slice_2d):
        slice_2d = np.nan_to_num(slice_2d)
        if np.max(slice_2d) == np.min(slice_2d):
            return np.zeros_like(slice_2d)
        norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
        return norm