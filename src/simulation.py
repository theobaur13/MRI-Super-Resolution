import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import fftshift, ifftshift, fftn, ifftn
from tqdm import tqdm
from src.utils import *

def generate_simulated_images(paths, dataset_type, axis):
    real_images = []
    real_kspaces = []
    simulated_kspaces = []
    simulated_images = []

    for path in tqdm(paths):
        if dataset_type == "brats":
            image = read_nifti(path)
        elif dataset_type == "pccai":
            image = read_metaimage(path)
        elif dataset_type == "ixi":
            image = read_nifti(path)

        kspace = convert_to_kspace(image)
        simulated_kspace = radial_undersampling(kspace, axis=axis, factor=0.4)
        # simulated_kspace = random_undersampling(simulated_kspace, factor=1.05)
        simulated_kspace = downsize_kspace(simulated_kspace, axis=axis, size=128)
        simulated_image = convert_to_image(simulated_kspace)
        
        real_images.append(image)
        real_kspaces.append(kspace)
        simulated_kspaces.append(simulated_kspace)
        simulated_images.append(simulated_image)

    index = 9
    display_simulated_comparison(real_images[index],
            simulated_images[index],
            real_kspaces[index],
            simulated_kspaces[index],
            axis=axis,
            highlight=True,
            show_kspace=True,
            show_image=True,
    )

def convert_to_kspace(image):
    kspace = fftshift(fftn(ifftshift(image), axes=(0, 1, 2)))
    kspace /= jnp.sqrt(kspace.shape[0] * kspace.shape[1] * kspace.shape[2])
    return kspace

def convert_to_image(kspace):
    image = fftshift(ifftn(ifftshift(kspace), axes=(0, 1, 2)))
    image *= jnp.sqrt(kspace.shape[0] * kspace.shape[1] * kspace.shape[2])
    return image

def downsize_kspace(kspace, axis, size=256):
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

def random_undersampling(kspace, factor=1.2, seed=42):
    np.random.seed(seed)
    mask = np.random.choice([0, 1], size=kspace.shape, p=[1 - 1 / factor, 1 / factor])
    return kspace * mask

def cartesian_undersampling(kspace, axis, factor=3):
    # Create a mask that keeps every x-th line along the specified axis
    mask = jnp.ones(kspace.shape)
    slices = [slice(None)] * 3
    slices[(axis + 1) % 3] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 0
    return kspace * mask

def radial_undersampling(kspace, axis, factor=0.5):
    radius = int((kspace.shape[(axis + 1) % 3] * factor) // 2)
    radius = max(radius, 1)

    # Create a mask that keeps only the center of the k-space, where the axis is the axis of slicing
    mask = jnp.zeros(kspace.shape)
    center = jnp.array(kspace.shape) // 2
    Z, Y, X = jnp.indices(kspace.shape)

    if axis == 0:       # Cylinder along the z-axis
        mask = jnp.where(jnp.sqrt((X - center[2])**2 + (Y - center[1])**2) <= radius, 1, 0)
    elif axis == 1:     # Cylinder along the y-axis
        mask = jnp.where(jnp.sqrt((X - center[2])**2 + (Z - center[0])**2) <= radius, 1, 0)
    elif axis == 2:     # Cylinder along the x-axis
        mask = jnp.where(jnp.sqrt((Y - center[1])**2 + (Z - center[0])**2) <= radius, 1, 0)

    return kspace * mask

def variable_density_undersampling(kspace, factor=1.1, ks=30):
    # Chance of sampling a line is inversely proportional to its distance from the center
    center = jnp.array(kspace.shape) // 2
    Z, Y, X = jnp.indices(kspace.shape)
    distances = jnp.sqrt((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2)

    # Normalize distances to [0, 1]
    distances = distances / jnp.max(distances)

    # Sigmoid function
    distances = 1 / (1 + jnp.exp(-ks * (distances - 0.5)))

    # Flatten
    probabilities = 1 - distances
    probabilities = probabilities / factor
    probabilities = jnp.clip(probabilities, 0, 1)
    mask = (np.random.rand(*kspace.shape) < probabilities).astype(int)

    return kspace * mask