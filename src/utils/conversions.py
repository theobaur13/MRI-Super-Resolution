import torch
import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import fftshift, ifftshift, fftn, ifftn

def convert_to_kspace(image):
    kspace = fftshift(fftn(ifftshift(image), axes=(0, 1, 2)))
    kspace /= jnp.sqrt(kspace.shape[0] * kspace.shape[1] * kspace.shape[2])
    return kspace

def convert_to_image(kspace):
    image = fftshift(ifftn(ifftshift(kspace), axes=(0, 1, 2)))
    image *= jnp.sqrt(kspace.shape[0] * kspace.shape[1] * kspace.shape[2])
    return image

def jax_to_numpy(x):
    """Convert a JAX array to a NumPy array."""
    if isinstance(x, jnp.ndarray):
        return np.array(x)
    else:
        raise TypeError("Input must be a JAX array.")

def numpy_to_jax(x):
    """Convert a NumPy array to a JAX array."""
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    else:
        raise TypeError("Input must be a NumPy array.")

def convert_to_tensor(image_list, slice_axis):
    real_slices = []
    imag_slices = []

    for img in image_list:
        img_np = jnp.array(img)
        for i in range(img_np.shape[slice_axis]):
            slice_2d = img_np[i]
            real_slices.append(slice_2d.real)
            imag_slices.append(slice_2d.imag)
    
    real_slices = jnp.array(real_slices)
    imag_slices = jnp.array(imag_slices)

    real_tensor = torch.tensor(real_slices, dtype=torch.float32).unsqueeze(1)
    imag_tensor = torch.tensor(imag_slices, dtype=torch.float32).unsqueeze(1)
    return torch.cat((real_tensor, imag_tensor), dim=1)