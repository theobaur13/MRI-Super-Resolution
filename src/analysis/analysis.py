import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
from tqdm import tqdm
from src.utils.conversions import jax_to_numpy, numpy_to_jax

# Function to generate and display a brightness mask for a given slice in two hypervolumes
def generate_brightness_map(slices_1, slices_2, smoothing_factor=5):

    # Calculate mean intensity for each slice in the sets
    intensity_slice_1 = jnp.mean(slices_1, axis=0)
    intensity_slice_2 = jnp.mean(slices_2, axis=0)

    # Calculate the conversion mask
    conversion_slice = intensity_slice_1 / (intensity_slice_2 + 1e-8)
    low, high = jnp.percentile(conversion_slice, 1), jnp.percentile(conversion_slice, 90)           # Clip as the outer edges are not relevant
    conversion_slice = jnp.clip(conversion_slice, low, high)
    conversion_slice = ndimage.gaussian_filter(jax_to_numpy(conversion_slice), smoothing_factor)    # Smooth the mask

    # Plot map
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
    im1 = ax1.imshow(np.abs(intensity_slice_1), cmap="plasma")
    im2 = ax2.imshow(np.abs(intensity_slice_2), cmap="plasma")
    im3 = ax3.imshow(np.abs(conversion_slice), cmap="plasma")
    ax1.set_title("Slice 3T")
    ax2.set_title("Slice 1.5T")
    ax3.set_title("Conversion Mask")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    return numpy_to_jax(conversion_slice) 

# Function to calculate and compare the Signal-to-Noise Ratio (SNR) for two hypervolumes
def compare_snr(hypervolume_1_5T, hypervolume_3T, axis=0, x=30):
    # Extract top corner square of each 
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

# Calculate the average SNR of each pixel in a given slice across all samples
def generate_snr_map(slices, x=50):
    noise_patches = slices[:, :x, :x]
    noise_std = jnp.std(noise_patches) 
    signal_mean = jnp.mean(slices, axis=0)
    snr_map = signal_mean / (noise_std + 1e-8)
    return snr_map