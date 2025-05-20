import jax.numpy as jnp
from jax import random
from jax import lax

# This function performs random undersampling in k-space by randomly selecting a fraction of the data points to keep.
def random_undersampling(kspace, factor=1.2, seed=42):
    key = random.PRNGKey(seed)
    prob = 1 / factor
    mask = random.bernoulli(key, p=prob, shape=kspace.shape)
    return kspace * mask

# This function performs undersampling in k-space by keeping every x-th line along the specified axis.
def cartesian_undersampling(kspace, axis, factor=3):
    # Create a mask that keeps every x-th line along the specified axis
    mask = jnp.ones(kspace.shape)
    slices = [slice(None)] * 3
    slices[(axis + 1) % 3] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 0
    return kspace * mask

# This function creates a cylindrical mask in k-space, keeping only the center of the k-space.
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

# This function randomly samples lines in k-space with a probability that decreases with distance from the center.
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

def gaussian(volume, axis, sigma=0.5, mu=0.0, A=20, invert=False):
    # sigma: standard deviation of the Gaussian
    # mu: mean of the Gaussian
    # A: amplitude of the Gaussian
    shape = volume.shape
    coords = [jnp.linspace(0, 1, shape[i]) for i in range(3)]
    coords = coords[axis:] + coords[:axis]  
    X, Y, Z = jnp.meshgrid(*coords, indexing='ij')

    gaussian = A * jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / (2 * sigma ** 2))
    if invert:
        gaussian = A * jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / -(2 * sigma ** 2))

    inverse_permutation = jnp.argsort(jnp.array([axis, (axis + 1) % 3, (axis + 2) % 3]))
    gaussian = jnp.transpose(gaussian, axes=tuple(inverse_permutation))
    return gaussian

# This function magnifies the brightness/intensity of a volume such that the center of each image slice is magnified greater than the edges.
def gaussian_amplification(volume, axis, spread=0.5, centre=0.0, amplitude=20, invert=False):
    # Create a Gaussian map
    mask = gaussian(volume, axis, sigma=spread, mu=centre, A=amplitude, invert=invert)
    return volume * mask

# This function adds random noise to an image with a specified intensity and frequency.
def random_noise(image, key=42, intensity=0.1, frequency=0.1):
    key_obj = random.PRNGKey(key)
    key_mask, key_noise = random.split(key_obj)

    noise_mask = random.uniform(key_mask, shape=image.shape) < frequency
    random_noise = random.uniform(key_noise, shape=image.shape, minval=-intensity, maxval=intensity)
    noise = noise_mask * random_noise

    return image + noise

# This function applies noise gradually at a greater intensity to the edges of the image.
def rician_edge_noise(image, axis=2, base_noise=0.4, key=42, edge_strength=0.1):
    gaussian_mask = gaussian(image, axis=axis, sigma=base_noise, mu=0.5, A=1, invert=True)
    sigma_map = base_noise * edge_strength * gaussian_mask
    noisy_image = rician_noise(image, sigma_map, key=key)
    return noisy_image

# TODO: This function adds physiological noise to the image.
def physiological_noise():
    pass

# This function adds Rician noise to the image.
def rician_noise(image, base_noise, key=42):
    key_obj = random.PRNGKey(key)
    key_real, key_imag = random.split(key_obj)

    noise_real = random.normal(key_real, shape=image.shape) * base_noise
    noise_imag = random.normal(key_imag, shape=image.shape) * base_noise
    noisy_complex = image + noise_real + 1j * noise_imag
    noisy_image = jnp.abs(noisy_complex)
    return noisy_image

def partial_fourier(kspace, axis, fraction=0.625, phase_correction=None):
    # Truncation
    target_axis = (axis + 1) % 3
    N = kspace.shape[target_axis]
    remove = N - int(jnp.floor(N * fraction))
    mask = jnp.zeros(kspace.shape)
    mask = mask.at[(slice(None),) * target_axis + (slice(remove, N),)].set(1)

    if phase_correction == "homodyne":
        # TODO: Implement homodyne phase correction
        pass
    
    elif phase_correction == "PCOS":
        # TODO: Implement PCOS phase correction
        pass

    return kspace * mask

# TODO: Grey-white matter boundary contrast reduction.
def grey_white_matter_boundary_contrast_reduction():
    pass