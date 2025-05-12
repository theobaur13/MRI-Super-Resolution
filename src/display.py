import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from src.utils import jax_to_numpy, get_slice

def plot_slice_from_nifti(ax, nifti, slice=65, cmap='gray', axis=0):
    ax.imshow(jnp.abs(get_slice(nifti, slice, axis)), cmap=cmap)

def display_comparison(niftis, slice=24, axis=0):
    fig = plt.figure(figsize=(15, 10))

    for i, nifti in enumerate(niftis):
        ax = fig.add_subplot(1, len(niftis), i + 1)
        plot_slice_from_nifti(ax, nifti, slice=slice, axis=axis)
        ax.set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show(block=False)

def plot_3d_surfaces(volumes, slice_idx, axis=0, cmap="viridis", limit=0):
    fig = plt.figure(figsize=(15, 10))

    for i, volume in enumerate(volumes):
        ax = fig.add_subplot(1, len(volumes), i + 1, projection='3d')
        plot_surface(ax, volume, slice_idx, axis=axis, cmap=cmap, limit=limit)
        ax.set_title(f"K-space {i+1}")

    plt.tight_layout()
    plt.show(block=False)

def plot_surface(ax, slice, cmap="viridis", limit=0):
    numpy_slice = jax_to_numpy(slice)  # Convert JAX array to NumPy array

    X, Y = np.meshgrid(np.arange(numpy_slice.shape[1]), np.arange(numpy_slice.shape[0]))  # Create grid
    ax.plot_surface(X, Y, numpy_slice, cmap=cmap, edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity")
    ax.view_init(elev=30, azim=135)  # Adjust viewing angle
    if limit > 0:
        ax.set_zlim(0, limit)