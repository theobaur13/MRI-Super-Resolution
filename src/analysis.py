import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
from src.display import plot_surface

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

    return conversion_mask