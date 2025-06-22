import jax.numpy as jnp
from jax import random
import jax
from jax import lax
from src.utils.paths import get_seg_paths
from src.utils.readwrite import read_nifti

# This function performs undersampling in k-space by keeping every x-th line along the specified axis.
def cartesian_undersampling(kspace, axis, gap=4, spine_width=64):
    shape = kspace.shape
    target_axis = (axis + 1) % 3
    N = shape[target_axis]

    # Create 1D undersampling mask with 1s, set every gap-th to 0
    idx = jnp.arange(N)
    undersample_mask_1d = jnp.where((idx % gap) == 0, 0, 1)

    # Insert spine of 1s in the middle
    spine_start = (N - spine_width) // 2
    spine_end = spine_start + spine_width
    spine_mask = jnp.where((idx >= spine_start) & (idx < spine_end), 1, 0)

    # Combine: take max so spine overwrites undersampling
    final_mask_1d = jnp.maximum(undersample_mask_1d, spine_mask)

    # Reshape to broadcast along 3D volume
    broadcast_shape = [1, 1, 1]
    broadcast_shape[target_axis] = N
    final_mask = final_mask_1d.reshape(broadcast_shape)

    # Broadcast and apply mask
    return kspace * final_mask

# This function simulates a GRAPPA reconstruction. Since GRAPPA requires multiple coils we need to use an interpolation method.
def reconstruct_cartesian(kspace, axis, spine_width=64, kernel_size=3, multiplier=1000.0):
    shape = kspace.shape
    target_axis = (axis + 1) % 3
    N = shape[target_axis]

    # === Step 1: Extract ACS ===
    acs_start = (N - spine_width) // 2
    acs_end = acs_start + spine_width
    acs_slice = [slice(None)] * 3
    acs_slice[target_axis] = slice(acs_start, acs_end)
    acs = kspace[tuple(acs_slice)]  # shape e.g. [H, 24, D] if axis=1

    # === Step 2: Create training set from ACS ===
    pad = kernel_size // 2
    acs_padded = jnp.pad(acs, [(0, 0), (0, 0), (0, 0)], mode='constant')
    acs_padded = jnp.moveaxis(acs_padded, target_axis, 0)  # Move undersampled axis to front

    X = []
    y = []
    for i in range(pad, acs_padded.shape[0] - pad):
        window = acs_padded[i - pad:i + pad + 1]  # shape [kernel_size, H, D]
        target = acs_padded[i]  # shape [H, D]
        if i % 2 == 1:  # Simulate odd-indexed lines as "missing"
            X.append(window.reshape(kernel_size, -1).T)  # shape [H*D, kernel_size]
            y.append(target.reshape(-1))                # shape [H*D]
    X = jnp.concatenate(X, axis=0)  # shape [N_samples, kernel_size]
    y = jnp.concatenate(y, axis=0)  # shape [N_samples]

    # === Step 3: Solve least squares: X @ w ≈ y ===
    w, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)  # shape [kernel_size]

    # === Step 4: Apply weights to reconstruct missing lines ===
    kspace_padded = jnp.pad(kspace, [(0, 0), (0, 0), (0, 0)], mode='constant')
    kspace_padded = jnp.moveaxis(kspace_padded, target_axis, 0)  # shape [N, ...]

    def reconstruct_line(i, ks):
        start = i - pad
        window = lax.dynamic_slice(ks, (start, 0, 0), (kernel_size, ks.shape[1], ks.shape[2]))
        window = window.reshape(kernel_size, -1).T
        prediction = window @ w
        prediction = prediction.reshape(ks.shape[1:]) * multiplier      # Multiply by multiplier if effect is too soft
        is_zero_line = jnp.all(ks[i] == 0)
        return jnp.where(is_zero_line, prediction, ks[i])

    ks_reconstructed = jax.vmap(
        lambda i: reconstruct_line(i, kspace_padded),
        in_axes=0
    )(jnp.arange(kspace_padded.shape[0]))

    # Move axis back to original position
    kspace_reconstructed = jnp.moveaxis(ks_reconstructed, 0, target_axis)
    return kspace_reconstructed

def radial_undersampling(kspace, axis, radius=100, spoke_num=100):
    shape = kspace.shape
    D, H, W = shape[axis], shape[(axis + 1) % 3], shape[(axis + 2) % 3]
    center = jnp.array([H // 2, W // 2])

    # Create 1D arrays for angles and radius
    angles = jnp.linspace(0, jnp.pi, spoke_num, endpoint=False)
    r = jnp.arange(-radius, radius)

    # Meshgrid of all spoke points
    dx = jnp.cos(angles)[:, None]
    dy = jnp.sin(angles)[:, None]

    # Shape: (spoke_num, 2*radius)
    x_coords = jnp.clip((center[1] + r * dx), 0, W - 1).astype(jnp.int32)
    y_coords = jnp.clip((center[0] + r * dy), 0, H - 1).astype(jnp.int32)

    # Flatten the indices
    x_flat = x_coords.reshape(-1)
    y_flat = y_coords.reshape(-1)

    # Create 2D mask
    mask_2d = jnp.zeros((H, W), dtype=jnp.int32)
    mask_2d = mask_2d.at[y_flat, x_flat].set(1)

    # Repeat mask along the target axis
    if axis == 0:
        mask_3d = jnp.stack([mask_2d] * D, axis=0)
    elif axis == 1:
        mask_3d = jnp.stack([mask_2d] * D, axis=1)
    elif axis == 2:
        mask_3d = jnp.stack([mask_2d] * D, axis=2)

    return kspace * mask_3d
    
def spiral_undersampling(kspace, axis, turns=50, samples=45000, p=3):
    D = kspace.shape[axis]
    H = kspace.shape[(axis + 1) % 3]
    W = kspace.shape[(axis + 2) % 3]
    center = jnp.array([H // 2, W // 2])

    theta = jnp.linspace(0, 2 * jnp.pi * turns, samples)
    a = min(H, W) / (2 * theta[-1] ** p)
    r = a * theta ** p

    x = r * jnp.cos(theta) + center[1]
    y = r * jnp.sin(theta) + center[0]

    # Round to nearest integer pixel indices
    x = jnp.clip(jnp.round(x).astype(jnp.int32), 0, W - 1)
    y = jnp.clip(jnp.round(y).astype(jnp.int32), 0, H - 1)

    # Draw mask
    mask = jnp.zeros((H, W), dtype=jnp.int32)
    mask = mask.at[y, x].set(1)

    mask = jnp.stack([mask] * D, axis=axis)
    return kspace * mask

# This function randomly samples lines in k-space with a probability that decreases with distance from the center.
def variable_density_undersampling(kspace, density=0.5, steepness=15, seed=42):
    key = random.PRNGKey(seed)

    # Chance of sampling a line is inversely proportional to its distance from the center
    center = jnp.array(kspace.shape) // 2
    Z, Y, X = jnp.indices(kspace.shape)
    distances = jnp.sqrt((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2)

    # Normalize distances to [0, 1]
    distances = distances / jnp.max(distances)

    # Sigmoid function
    distances = 1 / (1 + jnp.exp(-steepness * (distances - 0.5)))

    # Flatten
    probabilities = 1 - distances
    probabilities = probabilities / density
    probabilities = jnp.clip(probabilities, 0, 1)
    mask = random.bernoulli(key, p=probabilities).astype(kspace.dtype)

    return kspace * mask

# This function creates a cylindrical mask in k-space, keeping only the center of the k-space.
def cylindrical_crop(kspace, axis, factor=0.5, edge_smoothing=0.5):
    radius = int((kspace.shape[(axis + 1) % 3] * factor) // 2)
    radius = max(radius, 1)
    
    center = jnp.array(kspace.shape) // 2
    Z, Y, X = jnp.indices(kspace.shape)
    
    if axis == 0:  # Cylinder along z-axis
        r = jnp.sqrt((X - center[2])**2 + (Y - center[1])**2)
    elif axis == 1:  # Cylinder along y-axis
        r = jnp.sqrt((X - center[2])**2 + (Z - center[0])**2)
    else:  # axis == 2, Cylinder along x-axis
        r = jnp.sqrt((Y - center[1])**2 + (Z - center[0])**2)
    
    # Normalize radius
    r_norm = r / radius

    # Tukey-like radial window: 1 in center, cosine taper in [1-alpha, 1], 0 outside
    def tukey(rn):
        return jnp.where(
            rn <= 1 - edge_smoothing,
            1.0,
            jnp.where(
                rn <= 1.0,
                0.5 * (1 + jnp.cos(jnp.pi * (rn - 1 + edge_smoothing) / edge_smoothing)),
                0.0
            )
        )

    tapered_mask = tukey(r_norm)
    return kspace * tapered_mask

def gaussian(volume, axis, sigma=0.5, mu=0.0, A=20.0):
    shape = volume.shape
    coords = [jnp.linspace(0, 1, shape[i]) for i in range(3)]

    # Rearrange coords so that the desired axis comes first
    coords = coords[axis:] + coords[:axis]
    X, Y, Z = jnp.meshgrid(*coords, indexing='ij')

    # Apply Gaussian
    g = A * jnp.exp(-((X - mu) ** 2 + (Y - mu) ** 2 + (Z - mu) ** 2) / (2 * sigma ** 2))

    # Transpose back to original axis order manually (static)
    if axis == 0:
        axes = (0, 1, 2)
    elif axis == 1:
        axes = (2, 0, 1)
    elif axis == 2:
        axes = (1, 2, 0)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    g = jnp.transpose(g, axes=axes)
    return g

# This function magnifies the brightness/intensity of a volume such that the center of each image slice is magnified greater than the edges.
def gaussian_amplification(volume, axis, spread=0.5, centre=0.0, amplitude=20, invert=False):
    # Create a Gaussian map
    mask = gaussian(volume, axis, sigma=spread, mu=centre, A=amplitude)
    
    if invert:
        mask = 1.0 - mask
    else:
        mask = 1.0 + mask
    return volume * mask

# This function adds noise to the image according to matter type.
def matter_noise(image, path, base_noise, key=42):
    csf_path, gm_path, wm_path = get_seg_paths(path)

    # Read the segmentation files
    csf_prob = read_nifti(csf_path).get_fdata()
    gm_prob = read_nifti(gm_path).get_fdata()
    wm_prob = read_nifti(wm_path).get_fdata()
    bg_prob = 1 - (csf_prob + gm_prob + wm_prob)

    temp_image = jnp.array(image)

    # Apply Gaussian noise to white matter
    wm_noise = gaussian_noise(temp_image, base_noise=base_noise, key=key)

    # Apply Rician noise to grey matter and CSF
    gm_noise = rician_noise(temp_image, base_noise=base_noise, key=key)
    csf_noise = rician_noise(temp_image, base_noise=base_noise, key=key)

    # Apply Rayleigh noise to background
    bg_noise = rayleigh_noise(temp_image, base_noise=base_noise*0.66, key=key)
    
    # Combine the noise according to the probabilities
    noisy_image = (
        wm_prob * wm_noise +
        gm_prob * gm_noise +
        csf_prob * csf_noise +
        bg_prob * bg_noise
    )

    return noisy_image

# This function adds Rician noise to the image.
def rician_noise(image, base_noise, key=42):
    key_obj = random.PRNGKey(key)
    key_real, key_imag = random.split(key_obj)

    signal_real = jnp.real(image)
    signal_imag = jnp.imag(image)

    noise_real = random.normal(key_real, shape=image.shape) * base_noise
    noise_imag = random.normal(key_imag, shape=image.shape) * base_noise

    noisy_real = signal_real + noise_real
    noisy_imag = signal_imag + noise_imag

    noisy_image = jnp.sqrt(noisy_real**2 + noisy_imag**2)
    return noisy_image

# This function adds Gaussian noise to the image.
def gaussian_noise(image, base_noise, key=42):
    key_obj = random.PRNGKey(key)
    noise = random.normal(key_obj, shape=image.shape) * base_noise
    return image + noise

def rayleigh_noise(image, base_noise, key=42):
    key_obj = random.PRNGKey(key)
    noise = random.rayleigh(key_obj, scale=base_noise, shape=image.shape)
    return image + noise

def partial_fourier(kspace, axis, fraction=0.625):
    target_axis = (axis + 2) % 3
    shape = kspace.shape
    N = shape[target_axis]
    
    # Compute how many elements to keep
    keep = int(N * fraction)
    
    # Create mask: ones in the last `keep` entries along the target axis
    idx = jnp.arange(N)
    mask_1d = (idx >= (N - keep)).astype(kspace.dtype)  # shape: (N,)
    
    # Reshape for broadcasting
    mask_shape = [1, 1, 1]
    mask_shape[target_axis] = N
    mask = jnp.reshape(mask_1d, mask_shape)

    # Broadcast to kspace shape and apply mask
    return kspace * mask

# This function reconstructs the truncated k-space (from partial_fourier) using Hermitian symmetry.
def hermitian_reconstruct(kspace, axis):
    target_axis = (axis + 2) % 3

    # Flip and conjugate along the target axis
    kspace_conj = jnp.conj(jnp.flip(kspace, axis=target_axis))

    # Create keep mask based on non-zero values in truncated k-space
    keep_mask = jnp.abs(kspace) > 0

    # Fill in missing values with Hermitian conjugates
    reconstructed = jnp.where(keep_mask, kspace, kspace_conj)
    return reconstructed

# Adjust the contrast of the image in accordance to GM, WM, CSF.
def matter_contrast(image, path, intensity=0.5):
    csf_path, gm_path, wm_path = get_seg_paths(path)

    # Read the segmentation files
    csf_prob = read_nifti(csf_path).get_fdata()
    gm_prob = read_nifti(gm_path).get_fdata()
    wm_prob = read_nifti(wm_path).get_fdata()

    gm_T1 = 1.62    # T1 relaxation time for grey matter is 62% longer at 3T than at 1.5T
    wm_T1 = 1.42    # T1 relaxation time for white matter is 42% longer at 3T than at 1.5T
    csf_T1 = 1.0

    gm_scale = 1 / gm_T1
    wm_scale = 1 / wm_T1
    csf_scale = 1 / csf_T1

    # Apply the scaling factors to the image
    adjusted_image = (
        image * gm_prob * gm_scale +
        image * wm_prob * wm_scale +
        image * csf_prob * csf_scale
    )

    image = (1 - intensity) * image + intensity * adjusted_image

    return (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image) + 1e-8)