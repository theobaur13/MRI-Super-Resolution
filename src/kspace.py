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

def crop(kspace, axis, size=256):
    # Crop the k-space to the specified size along the specified axis, keeping the aspect ratio
    center = np.array(kspace.shape) // 2
    
    # Create a mask that keeps only the center of the k-space, where the axis is the axis that is kept
    slices = [slice(None)] * 3
    if axis == 0:       # Cylinder along the z-axis
        longer_axis = 1 if kspace.shape[1] > kspace.shape[2] else 2
        shorter_axis = 2 if longer_axis == 1 else 1
    elif axis == 1:     # Cylinder along the y-axis
        longer_axis = 0 if kspace.shape[0] > kspace.shape[2] else 2
        shorter_axis = 2 if longer_axis == 0 else 0
    elif axis == 2:     # Cylinder along the x-axis
        longer_axis = 0 if kspace.shape[0] > kspace.shape[1] else 1
        shorter_axis = 1 if longer_axis == 0 else 0

    aspect_ratio = kspace.shape[longer_axis] / kspace.shape[shorter_axis]
    slices[longer_axis] = slice(center[longer_axis] - size // 2, center[longer_axis] + size // 2)
    slices[shorter_axis] = slice(center[shorter_axis] - int(size / aspect_ratio) // 2, center[shorter_axis] + int(size / aspect_ratio) // 2)
    return kspace[tuple(slices)]

def gaussian_plane(kspace, axis, sigma=0.5, mu=0.0, A=20, invert=False):
    x = jnp.linspace(0, 1, kspace.shape[axis])
    y = jnp.linspace(0, 1, kspace.shape[(axis + 1) % 3])
    z = jnp.linspace(0, 1, kspace.shape[(axis + 2) % 3])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    gaussian = A * jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / (2 * sigma ** 2))
    if invert:
        gaussian = jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / -(2 * sigma ** 2)) - 0.5
    return kspace * gaussian

def random_noise(image, intensity=0.1, frequency=0.1):
    noise_mask = np.random.rand(*image.shape) < frequency
    random_noise = np.random.uniform(-intensity, intensity, size=image.shape)
    noise = noise_mask * random_noise

    return image + noise

def local_subvoxel_shift():
    pass  # Placeholder for local subvoxel shift function