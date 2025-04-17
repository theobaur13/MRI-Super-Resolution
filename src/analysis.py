import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
from src.display import plot_surface
from tqdm import tqdm

def smooth_image(image, sigma=5):
    return ndimage.gaussian_filter(image, sigma=sigma)

def find_mean_intesity(set, axis=0):
    # Calculate mean intensity of the specified slice across all images in the set
    return np.mean(set, axis=axis)

def generate_brightness_mask(set_1, set_2, slice_idx, axis=0, sigma=5):
    # Calculate mean intensity for each slice in the sets
    intensity_cube_1 = find_mean_intesity(set_1, axis)
    intensity_cube_2 = find_mean_intesity(set_2, axis)

    # Calculate the conversion mask
    conversion_mask = intensity_cube_1 / intensity_cube_2
    conversion_mask = np.clip(conversion_mask, 0, 1)  # Clip values to [0, 1] range
    conversion_mask = smooth_image(conversion_mask, sigma)  # Smooth the mask

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    plot_surface(ax1, intensity_cube_1, slice_idx, axis=axis, cmap="plasma", limit=0)
    plot_surface(ax2, intensity_cube_2, slice_idx, axis=axis, cmap="plasma", limit=0)
    plot_surface(ax3, conversion_mask, slice_idx, axis=axis, cmap="inferno", limit=0)
    
    ax1.set_title("Mean Intensity Image 1")
    ax2.set_title("Mean Intensity Image 2")
    ax3.set_title("Conversion Mask")
    return conversion_mask

def compare_snr(scans_1_5T, scans_3T, axis=0, x=30):
    if axis == 0:
        noise_1_5T = scans_1_5T[:, :, 15:x+15, 15:x+15]
        noise_3T = scans_3T[:, :, 15:x+15, 15:x+15]
    elif axis == 1:
        noise_1_5T = scans_1_5T[:, 0:x, :, 0:x]
        noise_3T = scans_3T[:, 0:x, :, 0:x]
    elif axis == 2:
        noise_1_5T = scans_1_5T[:, 0:x, 0:x, :]
        noise_3T = scans_3T[:, 0:x, 0:x, :]

    snr_1_5T = np.zeros((scans_1_5T.shape[0], scans_1_5T.shape[axis + 1]))
    snr_3T = np.zeros((scans_1_5T.shape[0], scans_3T.shape[axis + 1]))

    for i in tqdm(range(scans_1_5T.shape[0])):
        for j in range(scans_1_5T.shape[axis + 1]):
            if axis == 0:
                slice_1_5T = scans_1_5T[i, j, :, :]
                slice_3T = scans_3T[i, j, :, :]
                noise_1_5T_slice = noise_1_5T[i, j, :, :]
                noise_3T_slice = noise_3T[i, j, :, :]
                
            elif axis == 1:
                slice_1_5T = scans_1_5T[i, :, j, :]
                slice_3T = scans_3T[i, :, j, :]
                noise_1_5T_slice = noise_1_5T[i, :, j, :]
                noise_3T_slice = noise_3T[i, :, j, :]

            elif axis == 2:
                slice_1_5T = scans_1_5T[i, :, :, j]
                slice_3T = scans_3T[i, :, :, j]
                noise_1_5T_slice = noise_1_5T[i, :, :, j]
                noise_3T_slice = noise_3T[i, :, :, j]

            # Avoid division by zero
            noise_std_1_5T = np.std(noise_1_5T_slice)
            noise_std_3T = np.std(noise_3T_slice)

            snr_1_5T[i, j] = np.mean(slice_1_5T) / noise_std_1_5T if noise_std_1_5T > 0 else 0
            snr_3T[i, j] = np.mean(slice_3T) / noise_std_3T if noise_std_3T > 0 else 0

    mean_snr_1_5T = np.mean(snr_1_5T, axis=axis)
    mean_snr_3T = np.mean(snr_3T, axis=axis)

    print("Mean SNR 1.5T: ", mean_snr_1_5T)
    print("Mean SNR 3T: ", mean_snr_3T)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mean_snr_1_5T, label='Mean 1.5T SNR', color='blue')
    ax.plot(mean_snr_3T, label='Mean 3T SNR', color='red')
    ax.set_xlabel('Slice index')
    ax.set_ylabel('SNR')
    ax.set_title('Mean Slice-Wise SNR')
    ax.legend()