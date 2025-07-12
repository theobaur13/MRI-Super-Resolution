import jax.numpy as jnp
from jax import random
import jax
from jax import lax

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