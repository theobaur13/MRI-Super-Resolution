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