import jax.numpy as jnp
from jax import random

def random_undersampling(kspace, factor=1.2, seed=42):
    key = random.PRNGKey(seed)
    prob = 1 / factor
    mask = random.bernoulli(key, p=prob, shape=kspace.shape)
    return kspace * mask

def cartesian_undersampling(kspace, axis, factor=3):
    # Create a mask that keeps every x-th line along the specified axis
    mask = jnp.ones(kspace.shape)
    slices = [slice(None)] * 3
    slices[(axis + 1) % 3] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 0
    return kspace * mask

def cylindrical_crop(kspace, axis, factor=0.5):
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

def variable_density_undersampling(kspace, factor=1.1, ks=30, seed=42):
    key = random.PRNGKey(seed)

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
    mask = random.bernoulli(key, p=probabilities).astype(kspace.dtype)

    return kspace * mask

def crop(kspace, axis, size=256):
    # Crop the k-space to the specified size along the specified axis, keeping the aspect ratio
    center = jnp.array(kspace.shape) // 2
    
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

def gaussian_amplification(volume, axis, sigma=0.5, mu=0.0, A=20, invert=False):
    x = jnp.linspace(0, 1, volume.shape[axis])
    y = jnp.linspace(0, 1, volume.shape[(axis + 1) % 3])
    z = jnp.linspace(0, 1, volume.shape[(axis + 2) % 3])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    gaussian = A * jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / (2 * sigma ** 2))
    if invert:
        gaussian = jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / -(2 * sigma ** 2)) - 0.5
    return volume * gaussian

def random_noise(image, key=42, intensity=0.1, frequency=0.1):
    key_obj = random.PRNGKey(key)
    key_mask, key_noise = random.split(key_obj)

    noise_mask = random.uniform(key_mask, shape=image.shape) < frequency
    random_noise = random.uniform(key_noise, shape=image.shape, minval=-intensity, maxval=intensity)
    noise = noise_mask * random_noise

    return image + noise