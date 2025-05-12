import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
from tqdm import tqdm
from src.utils import jax_to_numpy, numpy_to_jax
from src.display import plot_surface

# Function to generate and display a brightness mask for a given slice in two hypervolumes
def generate_brightness_mask(slices_1, slices_2, axis, sigma=5, lim=0):
    if axis == 2:
        slice_axis = 0
    elif axis == 1:
        slice_axis = 2
    elif axis == 0:
        slice_axis = 1

    # Calculate mean intensity for each slice in the sets
    intensity_slice_1 = jnp.mean(slices_1, axis=slice_axis)
    intensity_slice_2 = jnp.mean(slices_2, axis=slice_axis)

    # Calculate the conversion mask
    conversion_slice = intensity_slice_1 / intensity_slice_2
    conversion_slice = jnp.clip(conversion_slice, 0, 1)  # Clip values to [0, 1] range
    conversion_slice = ndimage.gaussian_filter(jax_to_numpy(conversion_slice), sigma=sigma)  # Smooth the mask
    conversion_slice = numpy_to_jax(conversion_slice)  # Convert back to JAX array

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    plot_surface(ax1, intensity_slice_1, cmap="plasma", limit=lim)
    plot_surface(ax2, intensity_slice_2, cmap="plasma", limit=lim)
    plot_surface(ax3, conversion_slice, cmap="inferno", limit=lim)
    
    ax1.set_title("Mean Intensity Image 1")
    ax2.set_title("Mean Intensity Image 2")
    ax3.set_title("Conversion Slice")
    return conversion_slice

# Function to calculate and compare the Signal-to-Noise Ratio (SNR) for two hypervolumes
def compare_snr(hypervolume_1_5T, hypervolume_3T, axis=0, x=30):
    # Extract noise regions
    if axis == 0:
        noise_1_5T = hypervolume_1_5T[:, :, 15:x+15, 15:x+15]
        noise_3T = hypervolume_3T[:, :, 15:x+15, 15:x+15]
    elif axis == 1:
        noise_1_5T = hypervolume_1_5T[:, 0:x, :, 0:x]
        noise_3T = hypervolume_3T[:, 0:x, :, 0:x]
    elif axis == 2:
        noise_1_5T = hypervolume_1_5T[:, 0:x, 0:x, :]
        noise_3T = hypervolume_3T[:, 0:x, 0:x, :]

    num_samples = hypervolume_1_5T.shape[0]
    num_slices = hypervolume_1_5T.shape[axis + 1]

    def compute_snr(vol, noise_vol, i, j):
        if axis == 0:
            signal_slice = vol[i, j, :, :]
            noise_slice = noise_vol[i, j, :, :]
        elif axis == 1:
            signal_slice = vol[i, :, j, :]
            noise_slice = noise_vol[i, :, j, :]
        elif axis == 2:
            signal_slice = vol[i, :, :, j]
            noise_slice = noise_vol[i, :, :, j]

        signal_mean = jnp.mean(signal_slice)
        noise_std = jnp.std(noise_slice)
        snr = jnp.where(noise_std > 0, signal_mean / noise_std, 0.0)
        return snr

    # Vectorized SNR computation using list comprehension
    snr_1_5T = jnp.array([[
        compute_snr(hypervolume_1_5T, noise_1_5T, i, j)
        for j in range(num_slices)]
        for i in tqdm(range(num_samples))])
    snr_3T = jnp.array([[
        compute_snr(hypervolume_3T, noise_3T, i, j)
        for j in range(num_slices)]
        for i in tqdm(range(num_samples))])

    # Mean SNR across all samples
    mean_snr_1_5T = jnp.mean(snr_1_5T, axis=0)
    mean_snr_3T = jnp.mean(snr_3T, axis=0)

    # Print
    print("Mean SNR 1.5T:", float(jnp.mean(mean_snr_1_5T)))
    print("Mean SNR 3T:", float(jnp.mean(mean_snr_3T)))

    # Plot (convert to numpy)
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(mean_snr_1_5T), label='Mean 1.5T SNR', color='blue')
    plt.plot(np.array(mean_snr_3T), label='Mean 3T SNR', color='red')
    plt.xlabel('Slice index')
    plt.ylabel('SNR')
    plt.title('Mean Slice-Wise SNR')
    plt.legend()
    plt.show()

def compare_gibbs_ringing(volume_1, volume_2, slice_idx, axis=0):
    if axis == 0:
        slice_1 = volume_1[slice_idx, :, :]
        slice_2 = volume_2[slice_idx, :, :]
    elif axis == 1:
        slice_1 = volume_1[:, slice_idx, :]
        slice_2 = volume_2[:, slice_idx, :]
    elif axis == 2:
        slice_1 = volume_1[:, :, slice_idx]
        slice_2 = volume_2[:, :, slice_idx]

    pass