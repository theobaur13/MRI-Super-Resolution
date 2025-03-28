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

def random_undersampling(kspace, factor=1.2):
    mask = jnp.random.choice([0, 1], size=kspace.shape, p=[1 - 1 / factor, 1 / factor])
    return kspace * mask

def cartesian_undersampling(kspace, axis, factor=3):
    # Create a mask that keeps every x-th line along the specified axis
    mask = jnp.ones(kspace.shape)
    slices = [slice(None)] * 3
    slices[(axis + 1) % 3] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 0
    return kspace * mask

def radial_undersampling(kspace, axis, radius=70):
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
    mask = (jnp.random.rand(*kspace.shape) < probabilities).astype(int)

    return kspace * mask