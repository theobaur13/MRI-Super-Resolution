import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from src.utils.readwrite import read_nifti
from src.simulation.pipeline import simluation_pipeline
import numpy as np
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
import torch
import torch.nn.functional as F

vgg_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

perceptual_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval()

def absolute_error(hr, lr):
    return np.abs(hr - lr)

def perceptual_loss(hr, lr):
    # Convert HR and LR slices to 3-channel tensors
    hr_tensor = torch.tensor(hr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    lr_tensor = torch.tensor(lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    # Normalize
    hr_tensor = vgg_normalize(hr_tensor[0]).unsqueeze(0)
    lr_tensor = vgg_normalize(lr_tensor[0]).unsqueeze(0)

    # Get feature maps
    with torch.no_grad():
        hr_feat = perceptual_vgg(hr_tensor)
        lr_feat = perceptual_vgg(lr_tensor)

    # Convert to numpy after detaching
    error_map = torch.abs(hr_feat - lr_feat).mean(dim=1)[0].detach().cpu().numpy()

    return error_map

def edge_loss(hr, lr):
    hr_tensor = torch.tensor(hr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    lr_tensor = torch.tensor(lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(hr_tensor.device)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(hr_tensor.device)

    gx = F.conv2d(lr_tensor, sobel_x, padding=1)
    gy = F.conv2d(lr_tensor, sobel_y, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    return grad.squeeze().detach().cpu().numpy()

def fourier_loss(hr, lr):
    hr_tensor = torch.tensor(hr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    lr_tensor = torch.tensor(lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    hr_fft = torch.fft.fft2(hr_tensor)
    lr_fft = torch.fft.fft2(lr_tensor)

    error_map = torch.abs(hr_fft - lr_fft).mean(dim=1)[0].detach().cpu().numpy()

    return error_map

def gram_matrix(features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

def style_loss(hr, lr):
    # Convert to tensors and add batch & channel dimensions
    hr = torch.tensor(hr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    lr = torch.tensor(lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Repeat to 3 channels (for VGG)
    lr = lr.repeat(1, 3, 1, 1)
    hr = hr.repeat(1, 3, 1, 1)

    # Normalize
    lr_norm = vgg_normalize(lr[0]).unsqueeze(0)
    hr_norm = vgg_normalize(hr[0]).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        lr_feat = perceptual_vgg(lr_norm)
        hr_feat = perceptual_vgg(hr_norm)

    # Gram matrices
    lr_gram = gram_matrix(lr_feat)
    hr_gram = gram_matrix(hr_feat)

    return torch.abs(lr_gram - hr_gram).squeeze()

path = "E:\\data-brats-2024\\BraSyn\\train\\BraTS-GLI-00000-000\\BraTS-GLI-00000-000-t2f.nii.gz"
axis = 2
slice = 72

hr_nifti = read_nifti(path)
lr_nifti = simluation_pipeline(hr_nifti, axis, visualize=False, slice=slice)

hr_vol = hr_nifti.get_fdata()
lr_vol = lr_nifti.get_fdata()

hr_slice = hr_vol.take(slice, axis=axis)
lr_slice = lr_vol.take(slice, axis=axis)

# Get the error map
error_map = style_loss(hr_slice, lr_slice)

# Display LR, error map, and HR
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(lr_slice, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('LR Slice')
axs[1].imshow(error_map, cmap='hot')
axs[1].set_title(f'Error Map')
axs[2].imshow(hr_slice, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('HR Slice')
plt.tight_layout()
plt.show()